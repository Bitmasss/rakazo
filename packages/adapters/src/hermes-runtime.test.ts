import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import type { IncomingHttpHeaders, IncomingMessage, ServerResponse } from "node:http";
import http from "node:http";
import type { AddressInfo } from "node:net";
import net from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AgentRunModel,
  AgentRunRequest,
  AgentRuntime,
  AgentRuntimeEvent,
  ConnectorTool,
} from "@rakazo/adapter-kit";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  HermesAgentRuntime,
  HermesToolBridge,
  hermesEndpoint,
  hostHermesProfiles,
} from "./hermes-runtime.js";

// A scriptable stand-in for Hermes' Agent API server. Each test sets `plan` before a run starts.

interface Recorded {
  method: string;
  path: string;
  headers: IncomingHttpHeaders;
  body: string;
}

interface FakePlan {
  runId: string;
  /** Events sent on GET /v1/runs/:id/events, as `data:` frames. */
  frames: unknown[];
  /** Keep the event stream open until the run is stopped or `finish` is called. */
  hold: boolean;
}

class FakeHermes {
  readonly requests: Recorded[] = [];
  plan: FakePlan = { runId: "hermes-run-1", frames: [], hold: false };
  url = "";
  private server = http.createServer((req, res) => {
    this.handle(req, res).catch(() => {
      res.destroy();
    });
  });
  private readonly held = new Set<ServerResponse>();

  async start() {
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }

  async stop() {
    for (const res of this.held) res.end();
    this.held.clear();
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }

  calls(method: string, pathSuffix: string) {
    return this.requests.filter((r) => r.method === method && r.path.endsWith(pathSuffix));
  }

  /** Sends the final frames on every held event stream and closes it. */
  finish(frames: unknown[]) {
    for (const res of this.held) {
      for (const f of frames) res.write(sseFrame(f));
      res.end();
    }
    this.held.clear();
  }

  private async handle(req: IncomingMessage, res: ServerResponse) {
    const body = await readAll(req);
    const path = (req.url ?? "").split("?")[0] as string;
    this.requests.push({ method: req.method ?? "", path, headers: req.headers, body });
    const route = /^(?:\/p\/[^/]+)?\/v1\/runs(?:\/([^/]+)\/(events|stop|steer|approval))?$/.exec(
      path,
    );
    if (!route) return json(res, 404, { error: "not found" });
    const action = route[2];
    if (!action) {
      if (req.method !== "POST") return json(res, 405, {});
      return json(res, 200, { run_id: this.plan.runId });
    }
    if (action === "events") {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      for (const f of this.plan.frames) res.write(sseFrame(f));
      if (this.plan.hold) this.held.add(res);
      else res.end();
      return;
    }
    if (action === "stop") {
      for (const held of this.held) held.end();
      this.held.clear();
    }
    return json(res, 200, {});
  }
}

const sseFrame = (data: unknown) => `data: ${JSON.stringify(data)}\n\n`;

function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readAll(req: IncomingMessage) {
  return new Promise<string>((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function freePort() {
  const server = net.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

async function drain(stream: AsyncIterable<AgentRuntimeEvent>) {
  const out: AgentRuntimeEvent[] = [];
  for await (const event of stream) out.push(event);
  return out;
}

async function waitFor(check: () => boolean, what: string) {
  const deadline = Date.now() + 5000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

async function mcp(link: { url: string; token: string }, body: unknown, token = link.token) {
  const res = await fetch(link.url, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, body: res.status === 202 ? undefined : ((await res.json()) as any) };
}

const done = (text: string): AgentRuntimeEvent => ({ type: "done", text });
const completed = (output: string) => ({
  event: "run.completed",
  output,
  usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
});

async function* baseEvents(): AsyncIterable<AgentRuntimeEvent> {
  yield done("from base");
}

function fakeBase() {
  const calls: AgentRunRequest[] = [];
  const runtime = {
    describe: () => ({ id: "base" }),
    run: (request: AgentRunRequest) => {
      calls.push(request);
      return baseEvents();
    },
    abort: async () => undefined,
  } as unknown as AgentRuntime;
  return { calls, runtime };
}

const TOOLS: ConnectorTool[] = [
  {
    name: "lookup",
    description: "Look something up",
    inputSchema: { type: "object", properties: { q: { type: "string" } } },
  },
  {
    name: "ask_user",
    description: "Ask the user to choose",
    inputSchema: { type: "object", properties: {} },
  },
];

let fake: FakeHermes;
let stateDir: string;
let bridge: HermesToolBridge | undefined;

beforeEach(async () => {
  fake = new FakeHermes();
  await fake.start();
  stateDir = mkdtempSync(join(tmpdir(), "hermes-runtime-test-"));
  bridge = undefined;
});

afterEach(async () => {
  bridge?.close();
  await fake.stop();
  rmSync(stateDir, { recursive: true, force: true });
});

function model(overrides: Partial<AgentRunModel> = {}): AgentRunModel {
  return {
    provider: "hermes",
    id: "default",
    baseUrl: `${fake.url}/v1`,
    apiKey: "k-1234567890",
    ...overrides,
  };
}

function request(overrides: Partial<AgentRunRequest> = {}): AgentRunRequest {
  return {
    botId: "bot-1",
    threadId: "thread-1",
    runId: "run-1",
    prompt: "Check the build.",
    instructions: "Be brief.",
    history: [],
    tools: [],
    model: model(),
    ...overrides,
  };
}

describe("HermesAgentRuntime routing", () => {
  it("sends a run whose provider is not hermes to the base runtime", async () => {
    const base = fakeBase();
    const runtime = new HermesAgentRuntime({ base: base.runtime });
    const req = request({ model: { provider: "openai", id: "gpt-x" } });

    const events = await drain(runtime.run(req));

    expect(events).toEqual([done("from base")]);
    expect(base.calls).toEqual([req]);
    expect(fake.requests).toEqual([]);
  });
});

describe("HermesAgentRuntime basic run", () => {
  it("streams text and usage, then finishes with the full reply", async () => {
    fake.plan = {
      runId: "hermes-run-1",
      frames: [
        { event: "message.delta", delta: "Hel" },
        { event: "message.delta", delta: "lo" },
        completed("Hello"),
      ],
      hold: false,
    };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    const events = await drain(runtime.run(request()));

    expect(events).toEqual([
      { type: "text", text: "Hel" },
      { type: "text", text: "lo" },
      {
        type: "usage",
        provider: "hermes",
        model: "default",
        inputTokens: 3,
        outputTokens: 4,
        totalTokens: 7,
        cacheReadTokens: null,
        costUsd: null,
        usageSource: "hermes",
      },
      done("Hello"),
    ]);
  });

  it("sends the session key, idempotency key, bearer key, instructions and input", async () => {
    fake.plan = { runId: "hermes-run-1", frames: [completed("ok")], hold: false };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    await drain(runtime.run(request({ runId: "run-42" })));

    const [post] = fake.calls("POST", "/v1/runs");
    expect(post?.path).toBe("/v1/runs");
    expect(post?.headers["x-hermes-session-key"]).toBe("rakazo:bot-1:thread-1");
    expect(post?.headers["idempotency-key"]).toBe("run-42");
    expect(post?.headers.authorization).toBe("Bearer k-1234567890");
    const body = JSON.parse(post?.body ?? "{}");
    expect(body.instructions).toContain("Be brief.");
    expect(body.instructions).toContain("Rakazo");
    expect(body.input).toContain("Check the build.");
  });

  it("routes a non-default profile to /p/<profile>/v1/runs", async () => {
    fake.plan = { runId: "hermes-run-1", frames: [completed("ok")], hold: false };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    const events = await drain(runtime.run(request({ model: model({ id: "fluence" }) })));

    expect(events.at(-1)).toEqual(done("ok"));
    expect(fake.calls("POST", "/v1/runs")[0]?.path).toBe("/p/fluence/v1/runs");
    expect(fake.calls("GET", "/v1/runs/hermes-run-1/events")[0]?.path).toBe(
      "/p/fluence/v1/runs/hermes-run-1/events",
    );
  });
});

describe("HermesAgentRuntime thread history", () => {
  it("sends unseen history once, then only messages Hermes has not seen", async () => {
    fake.plan = {
      runId: "hermes-run-1",
      frames: [{ event: "message.delta", delta: "ok" }, completed("ok")],
      hold: false,
    };
    const stateFile = join(stateDir, "hermes-state.json");
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime, stateFile });

    await drain(
      runtime.run(
        request({
          runId: "run-1",
          prompt: "first",
          history: [
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
          ],
        }),
      ),
    );
    await drain(
      runtime.run(
        request({
          runId: "run-2",
          prompt: "second",
          history: [
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
            { role: "user", content: "new message" },
          ],
        }),
      ),
    );

    const [first, second] = fake.calls("POST", "/v1/runs").map((r) => JSON.parse(r.body).input);
    expect(first).toBe(
      "Messages in this thread you haven't seen yet:\n\nUser: hi\n\nAssistant: hello\n\nfirst",
    );
    expect(second).toBe(
      "Messages in this thread you haven't seen yet:\n\nUser: new message\n\nsecond",
    );
  });
});

describe("HermesAgentRuntime tool bridge", () => {
  const SECRET = "s".repeat(32);

  async function startBridge() {
    const port = await freePort();
    bridge = new HermesToolBridge({ secret: SECRET, port, catalog: [] });
    return bridge;
  }

  const linkFor = (b: HermesToolBridge) =>
    b.link(hermesEndpoint(`${fake.url}/v1`, "default").profileKey);

  it("serves initialize, tools/list and tools/call to a running Hermes profile", async () => {
    const b = await startBridge();
    const executeTool = vi.fn(async () => ({ rows: 2 }));
    fake.plan = { runId: "hermes-run-1", frames: [], hold: true };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime, bridge: b });
    const events = drain(runtime.run(request({ tools: TOOLS, executeTool })));
    await waitFor(() => fake.calls("GET", "/events").length > 0, "the event stream");
    const link = linkFor(b);

    const init = await mcp(link, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-06-18" },
    });
    expect(init.status).toBe(200);
    expect(init.body.result.serverInfo.name).toBe("rakazo");
    expect(init.body.result.capabilities).toEqual({ tools: {} });

    const list = await mcp(link, { jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(list.body.result.tools.map((t: { name: string }) => t.name)).toEqual([
      "lookup",
      "ask_user",
    ]);
    expect(list.body.result.tools[0].inputSchema).toEqual(TOOLS[0]?.inputSchema);

    const call = await mcp(link, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "lookup", arguments: { q: "x" } },
    });
    expect(executeTool).toHaveBeenCalledWith("lookup", { q: "x" }, expect.any(String));
    expect(call.body.result).toEqual({
      content: [{ type: "text", text: JSON.stringify({ rows: 2 }) }],
    });

    fake.finish([completed("all done")]);
    const emitted = await events;
    expect(emitted).toContainEqual({
      type: "tool",
      name: "lookup",
      args: { q: "x" },
      executionId: expect.any(String),
    });
    expect(emitted.at(-1)).toEqual(done("all done"));
  });

  it("rejects a bridge call with the wrong token", async () => {
    const b = await startBridge();
    fake.plan = { runId: "hermes-run-1", frames: [], hold: true };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime, bridge: b });
    const events = drain(runtime.run(request({ tools: TOOLS })));
    await waitFor(() => fake.calls("GET", "/events").length > 0, "the event stream");
    const link = linkFor(b);

    const res = await mcp(link, { jsonrpc: "2.0", id: 1, method: "tools/list" }, "f".repeat(64));
    expect(res.status).toBe(401);

    fake.finish([completed("bye")]);
    await events;
  });

  it("ends the run with an ask event and stops Hermes when ask_user offers two options", async () => {
    const b = await startBridge();
    const executeTool = vi.fn();
    fake.plan = { runId: "hermes-run-1", frames: [], hold: true };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime, bridge: b });
    const events = drain(runtime.run(request({ tools: TOOLS, executeTool })));
    await waitFor(() => fake.calls("GET", "/events").length > 0, "the event stream");

    const res = await mcp(linkFor(b), {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: { name: "ask_user", arguments: { question: "Pick one", options: ["A", "B"] } },
    });
    expect(res.body.result.content[0].text).toBe(
      "Waiting for the user's choice. End your turn now.",
    );

    const emitted = await events;
    expect(emitted).toContainEqual({
      type: "ask",
      text: "Pick one",
      actions: [
        { id: "choice-1", label: "A" },
        { id: "choice-2", label: "B" },
      ],
    });
    expect(emitted.some((e) => e.type === "done")).toBe(false);
    expect(executeTool).not.toHaveBeenCalled();
    await waitFor(() => fake.calls("POST", "/stop").length > 0, "the stop call");
  });

  it("ends the run without done when a tool pauses for approval", async () => {
    const b = await startBridge();
    const executeTool = vi.fn(async () => ({
      kind: "agent_tool_result",
      content: [{ type: "text", text: "Waiting for approval." }],
      details: { approval: "paused" },
    }));
    fake.plan = { runId: "hermes-run-1", frames: [], hold: true };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime, bridge: b });
    const events = drain(runtime.run(request({ tools: TOOLS, executeTool })));
    await waitFor(() => fake.calls("GET", "/events").length > 0, "the event stream");

    const res = await mcp(linkFor(b), {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: { name: "lookup", arguments: { q: "risky" } },
    });
    expect(res.body.result.content[0].text).toBe(
      "Waiting for the user's approval or input. End your turn now.",
    );

    const emitted = await events;
    expect(executeTool).toHaveBeenCalledTimes(1);
    expect(emitted.some((e) => e.type === "done")).toBe(false);
    await waitFor(() => fake.calls("POST", "/stop").length > 0, "the stop call");
  });
});

describe("HermesAgentRuntime abort", () => {
  it("stops the Hermes run when the caller's signal aborts mid-run", async () => {
    fake.plan = { runId: "hermes-run-1", frames: [], hold: true };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });
    const controller = new AbortController();
    const outcome = drain(runtime.run(request(), { signal: controller.signal })).then(
      (events) => ({ events }),
      (error: unknown) => ({ error }),
    );
    await waitFor(() => fake.calls("GET", "/events").length > 0, "the event stream");

    controller.abort();

    await waitFor(
      () => fake.calls("POST", "/v1/runs/hermes-run-1/stop").length > 0,
      "the stop call",
    );
    expect(await outcome).toEqual({ events: expect.arrayContaining([done("stopped")]) });
  });
});

describe("HermesAgentRuntime steering", () => {
  it("finishes a completed run when the request can claim steering", async () => {
    fake.plan = { runId: "hermes-run-1", frames: [completed("finished")], hold: false };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });
    const claimSteering = vi.fn(async () => []);

    const outcome = await Promise.race([
      drain(runtime.run(request({ claimSteering }))),
      new Promise<"timed out">((resolve) => setTimeout(() => resolve("timed out"), 3000)),
    ]);
    // Stops the steering loop even when the run above never finished.
    await runtime.abort("run-1");

    expect(outcome).not.toBe("timed out");
    expect(outcome).toContainEqual(done("finished"));
  });
});

describe("HermesAgentRuntime Hermes events", () => {
  it("declines a Hermes host approval request and reports it as progress", async () => {
    fake.plan = {
      runId: "hermes-run-1",
      frames: [
        { event: "approval.request", request_id: "req-7", command: "rm -rf /tmp/scratch" },
        completed("skipped it"),
      ],
      hold: false,
    };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    const events = await drain(runtime.run(request()));

    const [approval] = fake.calls("POST", "/v1/runs/hermes-run-1/approval");
    expect(JSON.parse(approval?.body ?? "{}")).toEqual({ choice: "deny", request_id: "req-7" });
    expect(events).toContainEqual({
      type: "progress",
      text: "Declined a command on the Hermes host (rm -rf /tmp/scratch). Use the computer tools instead.",
    });
    expect(events.at(-1)).toEqual(done("skipped it"));
  });
});

describe("HermesAgentRuntime empty replies", () => {
  it("yields the empty-response fallback when Hermes returns no text", async () => {
    fake.plan = {
      runId: "hermes-run-1",
      frames: [{ event: "run.completed", output: "" }],
      hold: false,
    };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    const events = await drain(runtime.run(request({ emptyResponseText: "Nothing came back." })));

    expect(events).toEqual([
      { type: "text", text: "Nothing came back." },
      done("Nothing came back."),
    ]);
  });

  it("uses the default fallback text when none is given", async () => {
    fake.plan = {
      runId: "hermes-run-1",
      frames: [{ event: "run.completed", output: "" }],
      hold: false,
    };
    const runtime = new HermesAgentRuntime({ base: fakeBase().runtime });

    const events = await drain(runtime.run(request()));

    expect(events.at(-1)).toEqual(done("No response. Try again."));
  });
});

describe("hostHermesProfiles", () => {
  it("reads the profiles and keys the host shares, skipping bad entries", () => {
    const dir = mkdtempSync(join(tmpdir(), "hermes-host-"));
    try {
      mkdirSync(join(dir, "hermes"));
      writeFileSync(
        join(dir, "hermes", "hermes-profiles.json"),
        JSON.stringify({
          profiles: {
            default: "k-default-123",
            fluence: "k-fluence-456",
            "bad name": "k-x-12345678",
            short: "k",
          },
        }),
      );
      expect(hostHermesProfiles(dir).profiles).toEqual({
        default: "k-default-123",
        fluence: "k-fluence-456",
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
