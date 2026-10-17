; Inno Setup script for Murmur.
; Build the app first (pyinstaller packaging/murmur.spec), then compile this with ISCC.exe.
; Installs per-user (no admin) into %LOCALAPPDATA%\Programs\Murmur. User data (settings, API keys,
; meetings) lives in %LOCALAPPDATA%\Murmur\data and is kept on uninstall.

#define AppName "Murmur"
#define AppVersion "1.0.0"
#define AppPublisher "Murad Huseynov"
#define AppExe "Murmur.exe"

[Setup]
AppId={{B3F7B0B9-2C4E-4E2A-9E7D-4D1A8C0F51A1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=Murmur-Setup
SetupIconFile=..\brand\kit\favicon.ico
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern
Compression=lzma2/normal
SolidCompression=yes
DisableDirPage=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

; Parakeet (the optional CPU engine, +2.9 GB) is not bundled: whisper + parakeet together exceed the
; ~4.2 GB single-Setup.exe limit. It can be shipped as a separate add-on if needed.

[Files]
; The PyInstaller bundle (app code, Python runtime, dependencies, static UI).
Source: "..\dist\Murmur\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
; ffmpeg (import, mixdown, screen remux).
Source: "..\bin\ffmpeg.exe"; DestDir: "{app}\bin"; Flags: ignoreversion
; Required models for the default Whisper pipeline.
Source: "..\models\faster-whisper\*"; DestDir: "{app}\models\faster-whisper"; Flags: recursesubdirs ignoreversion
Source: "..\models\diarization\*"; DestDir: "{app}\models\diarization"; Flags: recursesubdirs ignoreversion
Source: "..\models\aec\*"; DestDir: "{app}\models\aec"; Flags: recursesubdirs ignoreversion
Source: "..\models\silero_vad.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
; Optional Parakeet engine.

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
