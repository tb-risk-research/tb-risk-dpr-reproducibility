Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

pythonPath = scriptDir & "\.venv\Scripts\python.exe"
launchScript = scriptDir & "\_launch_gui.py"
parentDir = fso.GetParentFolderName(scriptDir)

Set env = WshShell.Environment("Process")
env("PYTHONPATH") = parentDir

If Not fso.FileExists(pythonPath) Then
    MsgBox "Virtual environment not found!" & vbCrLf & "Please run the launcher first.", vbCritical, "TB Risk"
    WScript.Quit 1
End If

WshShell.Run """" & pythonPath & """ """ & launchScript & """", 1, True
