import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import type {
  AdapterContext,
  AgentRunRequest,
  AgentRuntime,
  AgentRuntimeEvent,
  AgentToolExecutionResult,
  ConnectorTool,
} from "@rakazo/adapter-kit";
import { HERMES_PROVIDER_ID } from "@rakazo/contracts";
import { getLogger } from "@rakazo/logging";
import { isToolPauseResult } from "./approval-effect.js";

// Hermes Agent as a bot's runtime. A bot whose model connection is Hermes runs each turn on a Hermes
// Agent API server (POST /v1/runs, then its event stream) instead of in Pi: Hermes keeps the bot's
// memory, skills and its own conversation, one per Rakazo thread. Rakazo's tools reach Hermes over
// MCP (HermesToolBridge), and every call still goes through the run's executeTool, so approvals,
// effects and connector routing behave as they do for Pi. Every other bot runs on the base runtime.

/** Where the worker serves Rakazo's tools to Hermes (loopback). */
export const HERMES_BRIDGE_PORT = 7095;
const RUN_TIMEOUT_MS = 30 * 60_000; // a stream silent this long is given up on
const STEER_POLL_MS = 1500;
const CONTEXT_MESSAGES = 30;
const CONTEXT_CHARS = 24_000;
const TOOL_RESULT_CHARS = 60_000;
// Messages already in Hermes' own conversation, remembered by content so they aren't sent twice.
const SEEN_LIMIT = 400;
const SENT_LIMIT = 12; // recent turns sent, kept whole: a user message shows up inside its turn's prompt
const SENT_CHARS = 8000;

const HERMES_NOTE = [
  "You are running inside Rakazo as this bot. Your final reply text is posted to the user's thread.",
  "Rakazo's tools are available to you over MCP from the server named rakazo (mcp__rakazo__*). Prefer them",
  "over your own equivalents for anything the user or other bots should see: message_user for short",
  "progress updates, ask_user for a multiple-choice question, request_takeover when the user must act on",
  "the computer, message_bot and handoff_to_bot for teammates, schedule_* for routines, and Rakazo's",
  "computer, browser, memory and integration tools. Rakazo asks the user before consequential actions;",
  "when a tool says it is waiting for the user, end your turn.",
].join(" ");

type Fetch = typeof fetch;

export interface HermesRuntimeOptions {
  /** Runs every bot that isn't on Hermes. */
  base: AgentRuntime;
  /** Gives Hermes this run's tools; without it Hermes works with its own tools only. */
  bridge?: HermesToolBridge;
  /** Where each Hermes conversation's read position is kept across restarts. */
  stateFile?: string;
  fetch?: Fetch;
}

interface ActiveHermesRun {
  stop: () => Promise<void>;
}

export class HermesAgentRuntime implements AgentRuntime {
  private readonly active = new Map<string, ActiveHermesRun>();
  private readonly conversations: ConversationStore;
  private readonly fetch: Fetch;

  constructor(private readonly options: HermesRuntimeOptions) {
    this.conversations = new ConversationStore(options.stateFile);
    this.fetch = options.fetch ?? fetch;
  }

  describe() {
    return this.options.base.describe();
  }

  async abort(runId: string): Promise<void> {
    const run = this.active.get(runId);
    if (run) await run.stop();
    else await this.options.base.abort(runId);
  }

  run(
    request: AgentRunRequest,
    context?: Partial<AdapterContext>,
  ): AsyncIterable<AgentRuntimeEvent> {
    if (request.model.provider !== HERMES_PROVIDER_ID)
      return this.options.base.run(request, context);
    return this.runOnHermes(request, context);
  }

  private async *runOnHermes(
    request: AgentRunRequest,
    context?: Partial<AdapterContext>,
  ): AsyncIterable<AgentRuntimeEvent> {
    const endpoint = hermesEndpoint(request.model.baseUrl, request.model.id, request.model.apiKey);
    const sessionKey = `rakazo:${request.botId}:${request.threadId}`;
    const queue = new EventQueue<AgentRuntimeEvent>();
    const controller = new AbortController();
    const signal = context?.signal
      ? AbortSignal.any([context.signal, controller.signal])
      : controller.signal;
    let hermesRunId: string | undefined;
    let paused = false;
    let sentText = "";

    const stopInHermes = async () => {
      if (!hermesRunId) return;
      await this.call(endpoint, `/v1/runs/${encodeURIComponent(hermesRunId)}/stop`, {
        method: "POST",
        body: "{}",
      }).catch(() => undefined);
    };
    // A tool asked the user or is waiting for their approval: Rakazo resumes the run later.
    const pause = (event?: AgentRuntimeEvent) => {
      if (event) queue.push(event);
      paused = true;
      void stopInHermes();
      queue.close();
    };
    this.active.set(request.runId, {
      stop: async () => {
        controller.abort();
        await stopInHermes();
      },
    });
    const endTools = this.options.bridge?.begin(endpoint.profileKey, {
      request,
      emit: (event) => queue.push(event),
      pause,
    });

    try {
      const conversation = this.conversations.get(sessionKey);
      const unseen = unseenHistory(request.history, conversation);
      const input = [
        unseen.length
          ? `Messages in this thread you haven't seen yet:\n\n${formatHistory(unseen)}`
          : "",
        request.prompt,
      ]
        .filter(Boolean)
        .join("\n\n");
      const instructions = [request.instructions, HERMES_NOTE].filter(Boolean).join("\n\n");
      const started = await this.call(endpoint, "/v1/runs", {
        method: "POST",
        headers: { "X-Hermes-Session-Key": sessionKey, "Idempotency-Key": request.runId },
        body: JSON.stringify({
          input,
          instructions,
          ...(request.model.thinkingLevel && request.model.thinkingLevel !== "off"
            ? {
                model_options: {
                  reasoning_effort:
                    request.model.thinkingLevel === "max" ? "xhigh" : request.model.thinkingLevel,
                },
              }
            : {}),
        }),
        signal,
      });
      const accepted = (await hermesJson(started)) as { run_id?: string; id?: string };
      hermesRunId = String(accepted.run_id ?? accepted.id ?? "");
      if (!hermesRunId) throw new Error("Hermes started no run");
      this.conversations.remember(sessionKey, request.history, input);
      if (signal.aborted) await stopInHermes();
      else signal.addEventListener("abort", () => void stopInHermes(), { once: true });

      // Your messages written while it works reach the run in Hermes as steering, until it ends.
      const streamEnded = new AbortController();
      const steering = this.steer(
        endpoint,
        hermesRunId,
        request,
        AbortSignal.any([signal, streamEnded.signal]),
      );
      const stream = this.follow(endpoint, hermesRunId, request, queue, signal, (text) => {
        sentText += text;
      });
      stream
        // Stopped by the caller: the stream breaking off is expected, not a failure.
        .catch((error: unknown) => (signal.aborted ? undefined : queue.fail(error)))
        .finally(() => {
          streamEnded.abort();
          queue.close();
        });

      for await (const event of queue) yield event;
      await steering.catch(() => undefined);
      if (paused) return;
      if (signal.aborted) {
        yield { type: "done", text: "stopped" };
        return;
      }
      const reply = sentText.trim();
      this.conversations.rememberReply(sessionKey, reply);
      if (!reply && !request.allowSilentEmpty) {
        const fallback = request.emptyResponseText ?? "No response. Try again.";
        yield { type: "text", text: fallback };
        yield { type: "done", text: fallback };
        return;
      }
      yield { type: "done", text: reply };
    } finally {
      endTools?.();
      this.active.delete(request.runId);
      if (!paused && !signal.aborted && hermesRunId && queue.failed) await stopInHermes();
    }
  }

  // The run's event stream, as Rakazo events.
  private async follow(
    endpoint: HermesEndpoint,
    runId: string,
    request: AgentRunRequest,
    queue: EventQueue<AgentRuntimeEvent>,
    signal: AbortSignal,
    onText: (text: string) => void,
  ) {
    const response = await this.call(endpoint, `/v1/runs/${encodeURIComponent(runId)}/events`, {
      headers: { Accept: "text/event-stream" },
      signal,
      timeoutMs: 0,
    });
    if (!response.ok || !response.body) await hermesJson(response);
    let ended = false;
    const idle = new IdleTimer(RUN_TIMEOUT_MS, () =>
      queue.fail(new Error("Hermes stopped sending events for this run")),
    );
    try {
      for await (const data of sseData(response.body as ReadableStream<Uint8Array>, () =>
        idle.kick(),
      )) {
        if (queue.closed) break;
        const event = data as Record<string, any>;
        switch (event.event) {
          case "message.delta": {
            const text = String(event.delta ?? "");
            if (text) {
              onText(text);
              queue.push({ type: "text", text });
            }
            break;
          }
          case "message.interim":
            if (event.text && !event.already_streamed)
              queue.push({ type: "progress", text: String(event.text) });
            break;
          case "tool.started": {
            const name = String(event.tool ?? "tool");
            // Rakazo's own tools show as Rakazo tool calls (the bridge emits those).
            if (!/^mcp__rakazo__/.test(name)) {
              queue.push({
                type: "progress",
                text: toolActivity(name, event.preview),
                activity: true,
              });
            }
            break;
          }
          case "subagent.start":
          case "subagent.complete":
            queue.push({
              type: "subagent",
              agentId: String(
                event.subagent_id ?? event.agent_id ?? event.task_id ?? `${runId}:subagent`,
              ),
              name: String(event.name ?? "Helper"),
              task: String(event.goal ?? event.task ?? ""),
              status:
                event.event === "subagent.start" ? "running" : event.error ? "failed" : "completed",
              ...(event.result ? { result: String(event.result) } : {}),
            });
            break;
          case "approval.request":
            // Commands Hermes wants to run on its own host aren't approved from Rakazo: decline, and
            // the bot is told to use Rakazo's computer (where the user approves actions) instead.
            await this.call(endpoint, `/v1/runs/${encodeURIComponent(runId)}/approval`, {
              method: "POST",
              body: JSON.stringify({
                choice: "deny",
                ...(event.request_id ? { request_id: event.request_id } : {}),
              }),
            }).catch(() => undefined);
            queue.push({
              type: "progress",
              text: `Declined a command on the Hermes host (${String(event.command ?? event.description ?? "a command").slice(0, 120)}). Use the computer tools instead.`,
            });
            break;
          case "run.completed": {
            ended = true;
            if (
              !request.allowSilentEmpty &&
              typeof event.output === "string" &&
              event.output &&
              !queue.sawText
            ) {
              onText(event.output);
              queue.push({ type: "text", text: event.output });
            }
            const usage = hermesUsage(event.usage, endpoint.profile);
            if (usage) {
              if (request.onUsage) {
                await request.onUsage(usage);
                queue.push({ ...usage, accounted: true });
              } else queue.push(usage);
            }
            break;
          }
          case "run.failed":
            ended = true;
            throw new Error(`Hermes: ${String(event.error ?? "the run failed")}`);
          case "run.cancelled":
          case "run.interrupted":
            ended = true;
            break;
        }
        if (ended) break;
      }
    } finally {
      idle.stop();
    }
    if (!ended && !signal.aborted && !queue.closed)
      throw new Error("The connection to Hermes closed before the run finished");
  }

  private async steer(
    endpoint: HermesEndpoint,
    runId: string,
    request: AgentRunRequest,
    signal: AbortSignal,
  ) {
    if (!request.claimSteering) return;
    const seen: string[] = [];
    while (!signal.aborted) {
      const items = await request.claimSteering(seen).catch(() => []);
      for (const item of items) {
        seen.push(item.id);
        await this.call(endpoint, `/v1/runs/${encodeURIComponent(runId)}/steer`, {
          method: "POST",
          body: JSON.stringify({ input: item.text }),
        }).catch((error: unknown) => getLogger().warn("hermes steer failed", { error }));
      }
      await sleep(STEER_POLL_MS, signal);
    }
  }

  private async call(
    endpoint: HermesEndpoint,
    pathname: string,
    init: RequestInit & { timeoutMs?: number } = {},
  ) {
    const { timeoutMs = 60_000, ...rest } = init;
    const signals = [rest.signal, timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined].filter(
      Boolean,
    ) as AbortSignal[];
    try {
      return await this.fetch(endpoint.url + pathname, {
        ...rest,
        redirect: "error",
        signal: signals.length ? AbortSignal.any(signals) : undefined,
        headers: {
          ...(endpoint.apiKey ? { Authorization: `Bearer ${endpoint.apiKey}` } : {}),
          ...(rest.body ? { "Content-Type": "application/json" } : {}),
          ...(rest.headers as Record<string, string> | undefined),
        },
      });
    } catch (error) {
      if (rest.signal?.aborted) throw error;
      throw new Error(`Could not reach Hermes at ${endpoint.url}: ${(error as Error).message}`);
    }
  }
}

// --- the endpoint a bot runs on ----------------------------------------------------------------

export interface HermesEndpoint {
  url: string; // the server, or /p/<profile> on it for a profile other than the default
  apiKey?: string;
  profile: string;
  profileKey: string; // stable id of this server + profile (the tool bridge's route)
}

// The connection's address plus the bot's model id, which is the Hermes profile ("default" or a name).
export function hermesEndpoint(
  baseUrl: string | undefined,
  profile: string,
  apiKey?: string,
): HermesEndpoint {
  if (!baseUrl?.trim()) throw new Error("This Hermes connection has no address");
  const root = baseUrl.trim().replace(/\/+$/, "").replace(/\/v1$/, "");
  const name = profile.trim() || "default";
  const url = name === "default" ? root : `${root}/p/${encodeURIComponent(name)}`;
  return { url, apiKey, profile: name, profileKey: hermesProfileKey(root, name) };
}

export const hermesProfileKey = (root: string, profile: string) =>
  createHash("sha256").update(`${root}\n${profile}`).digest("hex").slice(0, 24);

async function hermesJson(response: Response): Promise<unknown> {
  const text = await response.text();
  let body: any;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    body = undefined;
  }
  if (!response.ok) {
    if (response.status === 401) throw new Error("Hermes rejected the API key for this profile");
    throw new Error(
      `Hermes: ${body?.error?.message ?? body?.error ?? (text.slice(0, 200) || response.status)}`,
    );
  }
  return body;
}

function hermesUsage(
  raw: any,
  profile: string,
): Extract<AgentRuntimeEvent, { type: "usage" }> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  return {
    type: "usage",
    provider: HERMES_PROVIDER_ID,
    model: profile,
    inputTokens: num(raw.input_tokens ?? raw.prompt_tokens),
    outputTokens: num(raw.output_tokens ?? raw.completion_tokens),
    totalTokens: num(raw.total_tokens),
    cacheReadTokens: num(raw.cache_read_tokens ?? raw.cache_read_input_tokens),
    costUsd: num(raw.cost_usd ?? raw.estimated_cost_usd),
    usageSource: "hermes",
  };
}

const toolActivity = (name: string, preview: unknown) => {
  const label = name.replace(/^mcp__[a-z0-9_-]+__/i, "").replace(/_/g, " ");
  const detail =
    typeof preview === "string" && preview.trim() ? `: ${preview.trim().slice(0, 120)}` : "";
  return `${label}${detail}`;
};

// --- what Hermes' conversation already holds -------------------------------------------------------
// Hermes keeps its own conversation per thread, which has every turn Rakazo sent it and every reply.
// Other messages in the thread (a teammate's, a group member's, one posted by a tool) are passed in
// with the next turn. Messages are remembered by content, since history items may lack ids.

type HistoryItem = AgentRunRequest["history"][number];
const digest = (text: string) =>
  createHash("sha256").update(text.trim()).digest("hex").slice(0, 20);

function unseenHistory(history: HistoryItem[], conversation: Conversation | undefined) {
  const recent = history.filter((m) => m.role !== "system" && m.content.trim());
  // A new conversation gets the recent thread as context; after that, only what it hasn't seen.
  const fresh = conversation
    ? recent.filter(
        (m) =>
          !conversation.seen.has(digest(m.content)) &&
          !conversation.sent.some((s) => s.includes(m.content.trim())),
      )
    : recent;
  const out: HistoryItem[] = [];
  let size = 0;
  for (const m of fresh.slice(-CONTEXT_MESSAGES).reverse()) {
    if (size + m.content.length > CONTEXT_CHARS) break;
    out.unshift(m);
    size += m.content.length;
  }
  return out;
}

const formatHistory = (items: HistoryItem[]) =>
  items.map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.trim()}`).join("\n\n");

interface Conversation {
  seen: Set<string>;
  sent: string[];
}
interface StoredConversation {
  seen: string[];
  sent: string[];
}

class ConversationStore {
  private data: Record<string, StoredConversation> | undefined;

  constructor(private readonly file?: string) {}

  private all() {
    if (!this.data) {
      try {
        this.data = this.file ? JSON.parse(fs.readFileSync(this.file, "utf8")) : {};
      } catch {
        this.data = {};
      }
    }
    return this.data as Record<string, StoredConversation>;
  }

  get(key: string): Conversation | undefined {
    const c = this.all()[key];
    return c && Array.isArray(c.seen)
      ? { seen: new Set(c.seen), sent: Array.isArray(c.sent) ? c.sent : [] }
      : undefined;
  }

  // Everything in the thread now is in Hermes' conversation (sent with this turn, or before it).
  remember(key: string, history: HistoryItem[], input: string) {
    const c = this.all()[key] ?? { seen: [], sent: [] };
    for (const m of history)
      if (m.role !== "system" && m.content.trim()) c.seen.push(digest(m.content));
    c.sent = [...c.sent, input.slice(0, SENT_CHARS)].slice(-SENT_LIMIT);
    this.save(key, c);
  }

  rememberReply(key: string, reply: string) {
    if (!reply) return;
    const c = this.all()[key] ?? { seen: [], sent: [] };
    c.seen.push(digest(reply));
    this.save(key, c);
  }

  private save(key: string, c: StoredConversation) {
    this.all()[key] = { seen: [...new Set(c.seen)].slice(-SEEN_LIMIT), sent: c.sent };
    if (!this.file) return;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
      fs.writeFileSync(`${this.file}.tmp`, JSON.stringify(this.all()));
      fs.renameSync(`${this.file}.tmp`, this.file);
    } catch (error) {
      getLogger().warn("could not save Hermes conversation state", { error });
    }
  }
}

// --- the tool bridge: Rakazo's tools for Hermes, over MCP -------------------------------------
// One MCP endpoint per Hermes profile (/mcp/<profileKey>), each with its own bearer token, which
// Hermes is given in its config (mcp_servers.rakazo). A call reaches the run that profile is working
// on right now and goes through that run's executeTool. Hermes reads a server's tool list when it
// connects, so the list is every tool a run may have; a call to one this run doesn't have is refused.

interface BridgedRun {
  request: AgentRunRequest;
  emit: (event: AgentRuntimeEvent) => void;
  pause: (event?: AgentRuntimeEvent) => void;
}

export interface HermesToolBridgeOptions {
  /** Signs each profile's token; keep it stable so Hermes' config keeps working. */
  secret: string;
  port: number;
  host?: string;
  /** Tools listed before a profile's first run (Hermes may connect before then). */
  catalog: ConnectorTool[];
}

export class HermesToolBridge {
  private readonly runs = new Map<string, BridgedRun>();
  private readonly known = new Map<string, ConnectorTool[]>();
  private server: http.Server | undefined;
  private seq = 0;

  constructor(private readonly options: HermesToolBridgeOptions) {}

  /** The address and token to give a Hermes profile (its config's mcp_servers.rakazo). */
  link(profileKey: string) {
    return {
      url: `http://${this.options.host ?? "127.0.0.1"}:${this.options.port}/mcp/${profileKey}`,
      token: this.token(profileKey),
    };
  }

  begin(profileKey: string, run: BridgedRun): () => void {
    this.listen();
    this.runs.set(profileKey, run);
    this.known.set(profileKey, run.request.tools);
    return () => {
      if (this.runs.get(profileKey) === run) this.runs.delete(profileKey);
    };
  }

  close() {
    this.server?.close();
    this.server = undefined;
  }

  private token(profileKey: string) {
    return createHmac("sha256", this.options.secret)
      .update(`hermes-bridge:${profileKey}`)
      .digest("hex");
  }

  private listen() {
    if (this.server) return;
    this.server = http.createServer((req, res) => {
      this.handle(req, res).catch((error: unknown) => {
        getLogger().warn("hermes tool bridge request failed", { error });
        if (!res.headersSent) res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "internal error" }));
      });
    });
    this.server.on("error", (error) =>
      getLogger().error("hermes tool bridge could not listen", { error }),
    );
    this.server.listen(this.options.port, this.options.host ?? "127.0.0.1");
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse) {
    const match = /^\/mcp\/([a-f0-9]{24})$/.exec(req.url?.split("?")[0] ?? "");
    const send = (status: number, body?: unknown) => {
      res.writeHead(status, body === undefined ? {} : { "Content-Type": "application/json" });
      res.end(body === undefined ? undefined : JSON.stringify(body));
    };
    if (!match) return send(404, { error: "not found" });
    if (req.method !== "POST") return send(405);
    const profileKey = match[1] as string;
    const want = Buffer.from(this.token(profileKey));
    const given = Buffer.from(String(req.headers.authorization ?? "").replace(/^Bearer\s+/i, ""));
    if (given.length !== want.length || !timingSafeEqual(given, want))
      return send(401, { error: "unauthorized" });
    const raw = await readBody(req, 4 * 1024 * 1024);
    let body: any;
    try {
      body = JSON.parse(raw);
    } catch {
      return send(400, {
        jsonrpc: "2.0",
        id: null,
        error: { code: -32700, message: "Parse error" },
      });
    }
    const messages = Array.isArray(body) ? body : [body];
    const replies = [];
    for (const m of messages) {
      if (!m || typeof m !== "object" || m.id === undefined) continue; // a notification
      replies.push(
        await this.rpc(profileKey, m).then(
          (result) => ({ jsonrpc: "2.0", id: m.id, result }),
          (error: any) => ({
            jsonrpc: "2.0",
            id: m.id,
            error: { code: error.code ?? -32603, message: error.message },
          }),
        ),
      );
    }
    if (!replies.length) return send(202);
    send(200, Array.isArray(body) ? replies : replies[0]);
  }

  private async rpc(profileKey: string, m: any) {
    switch (m.method) {
      case "initialize":
        return {
          protocolVersion: m.params?.protocolVersion ?? "2025-06-18",
          capabilities: { tools: {} },
          serverInfo: { name: "rakazo", version: "1" },
        };
      case "ping":
        return {};
      case "tools/list": {
        const tools =
          this.runs.get(profileKey)?.request.tools ??
          this.known.get(profileKey) ??
          this.options.catalog;
        return {
          tools: tools.map((t) => ({
            name: t.name,
            description: t.description,
            inputSchema: objectSchema(t.inputSchema),
          })),
        };
      }
      case "tools/call":
        return this.callTool(profileKey, String(m.params?.name ?? ""), m.params?.arguments ?? {});
      default:
        throw Object.assign(new Error(`Method not found: ${m.method}`), { code: -32601 });
    }
  }

  private async callTool(profileKey: string, name: string, args: Record<string, unknown>) {
    const text = (t: string, isError = false) => ({
      content: [{ type: "text", text: t }],
      ...(isError ? { isError: true } : {}),
    });
    const run = this.runs.get(profileKey);
    if (!run)
      return text(
        "No Rakazo turn is running for you right now, so Rakazo's tools aren't available.",
        true,
      );
    const { request } = run;
    if (!request.tools.some((t) => t.name === name))
      return text(`${name} isn't available in this turn.`, true);
    const executionId = JSON.stringify(["hermes-tool", request.runId, this.seq++]);
    run.emit({ type: "tool", name, args, executionId });

    if (name === "request_takeover") {
      run.pause({ type: "takeover", reason: String(args.reason ?? "I need you on the screen.") });
      return text("The user has been asked to take over the screen. End your turn now.");
    }
    if (name === "ask_user") {
      const options = Array.isArray(args.options)
        ? args.options.map((o) => String(o).trim()).filter(Boolean)
        : [];
      if (options.length < 2 || options.length > 4 || new Set(options).size !== options.length) {
        return text("ask_user needs two to four different options.", true);
      }
      run.pause({
        type: "ask",
        text: String(args.question ?? "What should I use?"),
        actions: options.map((label, index) => ({ id: `choice-${index + 1}`, label })),
      });
      return text("Waiting for the user's choice. End your turn now.");
    }
    if (!request.executeTool) return text("Rakazo can't run tools for this turn.", true);

    const startedAt = Date.now();
    let result: unknown;
    let failure: unknown;
    try {
      result = await request.executeTool(name, args, executionId);
      if (!isToolPauseResult(result)) return toolContent(result);
      run.pause();
      return text("Waiting for the user's approval or input. End your turn now.");
    } catch (error) {
      failure = error;
      return text(`${name} failed: ${(error as Error).message}`, true);
    } finally {
      await Promise.resolve(
        request.onToolCompleted?.({
          name,
          executionId,
          durationMs: Date.now() - startedAt,
          ...(failure ? { error: failure } : { result }),
          ...(isToolPauseResult(result) ? { paused: true } : {}),
        }),
      ).catch(() => undefined);
    }
  }
}

function toolContent(result: unknown) {
  if (
    result &&
    typeof result === "object" &&
    (result as AgentToolExecutionResult).kind === "agent_tool_result"
  ) {
    const content = (result as AgentToolExecutionResult).content.map((c) =>
      c.type === "image"
        ? { type: "image", data: c.data, mimeType: c.mimeType }
        : { type: "text", text: c.text },
    );
    return { content };
  }
  let text: string;
  try {
    text = typeof result === "string" ? result : (JSON.stringify(result) ?? "ok");
  } catch {
    text = String(result);
  }
  if (text.length > TOOL_RESULT_CHARS) text = `${text.slice(0, TOOL_RESULT_CHARS)} […cut]`;
  const isError = !!result && typeof result === "object" && "error" in (result as object);
  return { content: [{ type: "text", text }], ...(isError ? { isError: true } : {}) };
}

// MCP wants an object schema at the top.
const objectSchema = (schema: Record<string, unknown>) =>
  schema && schema.type === "object"
    ? schema
    : { type: "object", properties: {}, ...(schema ?? {}) };

function readBody(req: http.IncomingMessage, limit: number) {
  return new Promise<string>((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("request too large"));
        req.destroy();
      } else chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

// --- small plumbing ----------------------------------------------------------------------------------

class EventQueue<T> implements AsyncIterable<T> {
  private items: T[] = [];
  private waiters: Array<() => void> = [];
  private error: unknown;
  closed = false;
  failed = false;
  sawText = false;

  push(item: T) {
    if (this.closed) return;
    if ((item as { type?: string }).type === "text") this.sawText = true;
    this.items.push(item);
    this.wake();
  }

  close() {
    this.closed = true;
    this.wake();
  }

  fail(error: unknown) {
    if (this.closed) return;
    this.failed = true;
    this.error = error;
    this.close();
  }

  private wake() {
    for (const w of this.waiters.splice(0)) w();
  }

  async *[Symbol.asyncIterator]() {
    while (true) {
      if (this.items.length) {
        yield this.items.shift() as T;
        continue;
      }
      if (this.closed) {
        if (this.failed) throw this.error;
        return;
      }
      await new Promise<void>((r) => this.waiters.push(r));
    }
  }
}

class IdleTimer {
  private timer: NodeJS.Timeout | undefined;

  constructor(
    private readonly ms: number,
    private readonly onIdle: () => void,
  ) {
    this.kick();
  }

  kick() {
    clearTimeout(this.timer);
    this.timer = setTimeout(this.onIdle, this.ms);
    this.timer.unref?.();
  }

  stop() {
    clearTimeout(this.timer);
  }
}

async function* sseData(body: ReadableStream<Uint8Array>, onChunk: () => void) {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body as unknown as AsyncIterable<Uint8Array>) {
    onChunk();
    buffer += decoder.decode(chunk, { stream: true });
    for (let cut = buffer.indexOf("\n\n"); cut >= 0; cut = buffer.indexOf("\n\n")) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");
      if (!data) continue;
      try {
        yield JSON.parse(data) as unknown;
      } catch {
        // not JSON: a keep-alive or comment
      }
    }
  }
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
