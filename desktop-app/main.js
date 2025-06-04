const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require('path');

let win;

app.whenReady().then(() => {
  win = new BrowserWindow({
    height: 280,
    width: 888,
    //titleBarStyle: 'hidden', 
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,  // For ipcRenderer in renderer
      contextIsolation: true,
      webSecurity: false // For loading local images 
    }
  });

  if (app.isPackaged) {
    win.loadFile("./gui/build/index.html");
  } else {
    win.loadURL("http://localhost:3000");
  }

  // Listen for the 'open-directory-dialog' event from the renderer
  ipcMain.on('open-directory-dialog', () => {
    dialog.showOpenDialog(win, {
      properties: ['openDirectory'], // Open directory picker
    }).then(result => {
      if (!result.canceled) {
        // Send the selected directory back to the renderer
        win.webContents.send('selected-directory', result.filePaths[0]);
      }
    }).catch(err => {
      console.error('Error:', err);
    });
  });
});