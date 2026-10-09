import { Trans, useLingui } from "@lingui/react/macro";
import type { ModelCredential } from "@rakazo/contracts";
import { HERMES_PROVIDER_ID } from "@rakazo/contracts";
import { Button, Input } from "@rakazo/ui-web";
import { useEffect, useState } from "react";
import { rpc } from "../lib/rpc";
import { errorText } from "../lib/user-error";

type ProfileRow = { name: string; key: string };

/**
 * A Hermes Agent server as a model connection: bots whose model is one of its profiles run their
 * turns on Hermes (hermes-runtime.ts), keeping Hermes' own memory and skills. The default profile
 * uses the server's API key; each other profile has its own.
 */
export function HermesConnectionSettings({ onChanged }: { onChanged?: () => void }) {
  const { t } = useLingui();
  const [credential, setCredential] = useState<ModelCredential | null>(null);
  const [profiles, setProfiles] = useState<string[]>([]);
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8642");
  const [apiKey, setApiKey] = useState("");
  const [rows, setRows] = useState<ProfileRow[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  async function load() {
    const [credentials, catalog] = await Promise.all([rpc.models.credentials(), rpc.models.list()]);
    const found = credentials.find((entry) => entry.provider === HERMES_PROVIDER_ID) ?? null;
    setCredential(found);
    if (found?.baseUrl) setBaseUrl(found.baseUrl.replace(/\/v1\/?$/, ""));
    setProfiles(
      catalog
        .filter((entry) => entry.provider === HERMES_PROVIDER_ID && entry.id !== "default")
        .map((entry) => entry.id),
    );
  }

  // Loaded once; saving and removing load it again.
  useEffect(() => {
    load().catch(() => undefined);
  }, []);

  async function save(next: { profileKeys?: Record<string, string> } = {}) {
    setSaving(true);
    setMessage(null);
    try {
      const profileKeys = {
        ...Object.fromEntries(
          rows.filter((row) => row.name.trim()).map((row) => [row.name.trim(), row.key.trim()]),
        ),
        ...next.profileKeys,
      };
      await rpc.models.connect({
        provider: HERMES_PROVIDER_ID,
        baseUrl: baseUrl.trim(),
        modelId: "default",
        ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
        ...(Object.keys(profileKeys).length ? { profileKeys } : {}),
      });
      setApiKey("");
      setRows([]);
      await load();
      onChanged?.();
      setMessage({
        ok: true,
        text: t`Saved. Pick a Hermes profile as a bot's model to run it on Hermes.`,
      });
    } catch (error) {
      setMessage({ ok: false, text: errorText(error) });
    } finally {
      setSaving(false);
    }
  }

  const ready = baseUrl.trim().length > 0 && (credential !== null || apiKey.trim().length >= 8);
  return (
    <section
      data-testid="hermes-connection"
      aria-labelledby="hermes-connection-title"
      className="mt-8 border-t border-border pt-5"
    >
      <h3 id="hermes-connection-title" className="text-[15px] font-medium text-foreground">
        <Trans>Hermes Agent</Trans>
      </h3>
      <p className="mt-1 text-[13px] leading-[1.5] text-muted-foreground">
        <Trans>
          Run bots on your own Hermes server. Each profile is a model: a bot set to it thinks with
          Hermes' memory and skills, and uses Rakazo's tools.
        </Trans>
      </p>
      <div className="mt-3 grid gap-2">
        <label htmlFor="hermes-address" className="grid gap-1 text-[13px]">
          <span className="text-muted-foreground">
            <Trans>API server address</Trans>
          </span>
          <Input
            id="hermes-address"
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.target.value)}
            placeholder="http://127.0.0.1:8642"
          />
        </label>
        <label htmlFor="hermes-key" className="grid gap-1 text-[13px]">
          <span className="text-muted-foreground">
            {credential?.hasKey ? (
              <Trans>API key (saved; enter a new one to replace it)</Trans>
            ) : (
              <Trans>API key (API_SERVER_KEY)</Trans>
            )}
          </span>
          <Input
            id="hermes-key"
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            autoComplete="off"
          />
        </label>
        {credential ? (
          <div className="grid gap-1 text-[13px]">
            <span className="text-muted-foreground">
              <Trans>Profiles</Trans>
            </span>
            <ul className="grid gap-1">
              <li>default</li>
              {profiles.map((name) => (
                <li key={name} className="flex items-center gap-2">
                  <span className="grow">{name}</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={saving}
                    onClick={() => save({ profileKeys: { [name]: "" } })}
                  >
                    <Trans>Remove</Trans>
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {rows.map((row, index) => (
          <div key={`${index}`} className="flex gap-2">
            <Input
              value={row.name}
              placeholder={t`Profile name`}
              onChange={(event) =>
                setRows((list) =>
                  list.map((r, i) => (i === index ? { ...r, name: event.target.value } : r)),
                )
              }
            />
            <Input
              type="password"
              value={row.key}
              placeholder={t`Its API key`}
              autoComplete="off"
              onChange={(event) =>
                setRows((list) =>
                  list.map((r, i) => (i === index ? { ...r, key: event.target.value } : r)),
                )
              }
            />
          </div>
        ))}
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setRows((list) => [...list, { name: "", key: "" }])}
          >
            <Trans>Add a profile</Trans>
          </Button>
          <Button size="sm" disabled={!ready || saving} onClick={() => save()}>
            {credential ? <Trans>Save</Trans> : <Trans>Connect Hermes</Trans>}
          </Button>
        </div>
        {message ? (
          <p
            className={`text-[13px] ${message.ok ? "text-muted-foreground" : "text-destructive"}`}
            role="status"
          >
            {message.text}
          </p>
        ) : null}
      </div>
    </section>
  );
}
