// Starts `langcheck lsp` and connects it to VS Code. All checking happens on the
// langcheck server; this extension only passes settings and the API key along.
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

const SECTION = "langcheck";
const API_KEY_SECRET = "langcheck.apiKey";

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand("langcheck.setApiKey", async () => {
      const key = await vscode.window.showInputBox({
        prompt: "API key for the langcheck server",
        password: true,
        ignoreFocusOut: true,
      });
      if (key) {
        await context.secrets.store(API_KEY_SECRET, key.trim());
        await restart(context);
      }
    }),
    vscode.commands.registerCommand("langcheck.clearApiKey", async () => {
      await context.secrets.delete(API_KEY_SECRET);
      await restart(context);
    }),
    vscode.commands.registerCommand("langcheck.restart", () => restart(context)),
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (event.affectsConfiguration(SECTION)) {
        await restart(context);
      }
    }),
  );
  await start(context);
}

export async function deactivate(): Promise<void> {
  await stop();
}

async function restart(context: vscode.ExtensionContext): Promise<void> {
  await stop();
  await start(context);
}

async function stop(): Promise<void> {
  const running = client;
  client = undefined;
  await running?.stop();
}

async function start(context: vscode.ExtensionContext): Promise<void> {
  const config = vscode.workspace.getConfiguration(SECTION);
  const serverUrl = config.get<string>("serverUrl", "").trim();
  if (!serverUrl) {
    void vscode.window.showInformationMessage(
      "langcheck: set langcheck.serverUrl to start checking.",
    );
    return;
  }
  // Settings go in the environment rather than on the command line, so the API key
  // never shows in the process list.
  const env: NodeJS.ProcessEnv = { ...process.env, LANGCHECK_SERVER__URL: serverUrl };
  const apiKey = await context.secrets.get(API_KEY_SECRET);
  if (apiKey) {
    env["LANGCHECK_SERVER__API_KEY"] = apiKey;
  }
  // No `transport`: naming stdio makes the client add `--stdio`, which `langcheck lsp` rejects;
  // leaving it out still uses stdio.
  const serverOptions: ServerOptions = {
    command: config.get<string>("command", "langcheck"),
    args: [...config.get<string[]>("args", []), "lsp"],
    options: { env },
  };
  const languages = config.get<string[]>("languages", ["markdown", "plaintext"]);
  const clientOptions: LanguageClientOptions = {
    documentSelector: languages.flatMap((language) => [
      { scheme: "file", language },
      { scheme: "untitled", language },
    ]),
  };
  client = new LanguageClient("langcheck", "langcheck", serverOptions, clientOptions);
  try {
    await client.start();
  } catch (error) {
    client = undefined;
    void vscode.window.showErrorMessage(`langcheck: could not start the language server: ${error}`);
  }
}
