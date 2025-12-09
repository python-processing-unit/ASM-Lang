const vscode = require('vscode');
const path = require('path');

function getWorkspaceRoot() {
    const ws = vscode.workspace.workspaceFolders;
    if (!ws || ws.length === 0) return undefined;
    return ws[0].uri.fsPath;
}

function resolveInterpreterPath() {
    const config = vscode.workspace.getConfiguration();
    const py = config.get('asmln.pythonPath');
    return py || 'python';
}

function ensureSaved(editor) {
    if (!editor) return Promise.resolve(true);
    if (!editor.document.isDirty) return Promise.resolve(true);
    return editor.document.save();
}

function runCommandInTerminal(command, cwd) {
    const term = vscode.window.createTerminal({ cwd: cwd });
    term.show(true);
    term.sendText(command);
}

function runFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showInformationMessage('No active editor to run');
        return;
    }
    ensureSaved(editor).then((ok) => {
        if (!ok) return;
        const filePath = editor.document.uri.fsPath;
        const root = getWorkspaceRoot() || path.dirname(filePath);
        const interp = resolveInterpreterPath();
        const asmlnPath = path.join(root, 'asmln.py');
        const cmd = `${interp} "${asmlnPath}" "${filePath}"`;
        runCommandInTerminal(cmd, root);
    });
}

function runRepl() {
    const root = getWorkspaceRoot() || process.cwd();
    const interp = resolveInterpreterPath();
    const asmlnPath = path.join(root, 'asmln.py');
    const cmd = `${interp} "${asmlnPath}"`;
    runCommandInTerminal(cmd, root);
}

/** @param {vscode.ExtensionContext} context */
function activate(context) {
    context.subscriptions.push(
        vscode.commands.registerCommand('asmln.runFile', runFile),
        vscode.commands.registerCommand('asmln.runRepl', runRepl)
    );
}

function deactivate() {}

module.exports = { activate, deactivate };
