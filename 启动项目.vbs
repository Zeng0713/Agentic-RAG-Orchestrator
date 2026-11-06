Option Explicit
Dim fso, shell, root, pythonw, entry
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then pythonw = fso.BuildPath(root, "venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonw) Then
    MsgBox "Python environment not found. Run the dependency installer in this folder first.", 16, "Agentic RAG"
    WScript.Quit 1
End If
entry = fso.BuildPath(root, "launcher.pyw")
shell.CurrentDirectory = root
shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & entry & Chr(34), 0, False
