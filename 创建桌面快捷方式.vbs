Option Explicit
Dim fso, shell, root, pythonw, entry, shortcutPath, link, quiet
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then pythonw = fso.BuildPath(root, "venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then
    MsgBox "Python environment not found. Run the dependency installer first.", 16, "Agentic RAG"
    WScript.Quit 1
End If
entry = Chr(34) & fso.BuildPath(root, "launcher.pyw") & Chr(34)
shortcutPath = fso.BuildPath(shell.SpecialFolders("Desktop"), "Agentic RAG Orchestrator.lnk")
If fso.FileExists(shortcutPath) Then
    Set link = shell.CreateShortcut(shortcutPath)
    If link.Arguments <> entry Then
        shortcutPath = fso.BuildPath(shell.SpecialFolders("Desktop"), "Agentic RAG Orchestrator (" & fso.GetFolder(root).Name & ").lnk")
    End If
End If
Set link = shell.CreateShortcut(shortcutPath)
link.TargetPath = pythonw
link.Arguments = entry
link.WorkingDirectory = root
link.IconLocation = fso.BuildPath(root, "assets\app.ico") & ",0"
link.Description = "Agentic RAG local workspace launcher"
link.Save
quiet = False
If WScript.Arguments.Count > 0 Then quiet = (WScript.Arguments(0) = "--quiet")
If quiet Then
    WScript.Echo shortcutPath
Else
    MsgBox "Desktop shortcut created: Agentic RAG Orchestrator", 64, "Agentic RAG"
End If
