const express = require('express');
const fs = require('fs');
const path = require("path")
const multer = require('multer'); // For file uploads
const { spawn } = require('child_process');

const app = express();

const prot = 8005;
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const name = Date.now() + '-' + file.originalname;
        cb(null, name);
    }
});
const upload = multer({ storage: storage });

//app.use(express.static(__dirname));
app.use(express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => {
    res.status(200).sendFile(path.join(__dirname, "public", "index.html"));
});

app.post("/", upload.single("report"), (req, res) => {
    res.status(200);
    //res.status(200).send("Post request received");
});


// testing Analyze API

app.get("/analyze", (req, res) => {
  const pythonProcess = spawn('python', ['python/analyze.py']);
  console.log("Python script started");

  let output = '';
  pythonProcess.stdout.on('data', data => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', data => {
    console.error(`Python error: ${data}`);
  });

  pythonProcess.on('close', code => {
    try {
      const result = JSON.parse(output);
      res.status(200).json(result);
      console.log(result);
      console.log("Python script finished with code", code);
    } catch (err) {
      res.status(500).json({ error: "Failed to parse Python output" });
    }

    // Clean up after response is sent
    const uploadsDir = path.join(__dirname, 'uploads');
    fs.readdir(uploadsDir, (err, files) => {
      if (err) {
        console.error('Error reading uploads directory:', err);
        return;
      }

      files.forEach(file => {
        const filePath = path.join(uploadsDir, file);
        fs.unlink(filePath, err => {
          if (err) {
            console.error(`Error deleting file ${file}:`, err);
          } else {
            console.log(`Deleted: ${file}`);
          }
        });
      });
    });
  });
});





app.listen(prot, () => {
    console.log(`Server is running on port http://localhost:${prot}`);
});
