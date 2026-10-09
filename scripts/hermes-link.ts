// Prints the config a Hermes profile needs to use Rakazo's tools while it runs a bot's turn:
// the worker's MCP bridge (hermes-runtime.ts) and that profile's token. Add it to the profile's
// config.yaml (Hermes picks up new MCP servers on its own after a minute or two).
//
//   pnpm hermes:link <hermes address> [profile...]
//   pnpm hermes:link http://127.0.0.1:8642 default fluence
import {
  HERMES_BRIDGE_PORT,
  HermesToolBridge,
  hermesEndpoint,
} from "../packages/adapters/src/hermes-runtime.js";

const [address, ...profiles] = process.argv.slice(2);
const secret = process.env.ENCRYPTION_KEY;
if (!address || !secret) {
  process.stderr.write(
    "Usage: pnpm hermes:link <hermes address> [profile...] (with the worker's ENCRYPTION_KEY in .env)\n",
  );
  process.exit(1);
}
const bridge = new HermesToolBridge({ secret, port: HERMES_BRIDGE_PORT, catalog: [] });
for (const profile of profiles.length ? profiles : ["default"]) {
  const { url, token } = bridge.link(hermesEndpoint(address, profile).profileKey);
  const file =
    profile === "default" ? "~/.hermes/config.yaml" : `~/.hermes/profiles/${profile}/config.yaml`;
  process.stdout.write(
    [
      `# ${profile}: ${file}`,
      "mcp_servers:",
      "  rakazo:",
      `    url: ${url}`,
      "    headers:",
      `      Authorization: Bearer ${token}`,
      "    timeout: 900",
      "    connect_timeout: 30",
      "    enabled: true",
      "",
    ].join("\n"),
  );
}
