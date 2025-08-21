const express = require('express');
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

app.get("/analyze", (req, res) => {
    //res.status(200).sendFile(path.join(__dirname, "public", "index.html"));
    // Run Python script
    const pythonProcess = spawn('python',['python/analyze.py']);

    let output = '';
    pythonProcess.stdout.on('data', data => {
        output += data.toString();
        //console.log(`Python output: ${data}`);
    });
    pythonProcess.stderr.on('data', data => {
        console.error(`Python error: ${data}`);
    });
    pythonProcess.on('close', code => {
        // if (code === 0) {
        //     res.send(`<pre>${output}</pre>`);
        // } else {
        //     res.status(500).send('Error processing image');
        // }
        try {
            const result = JSON.parse(output);
            res.status(200).json(result); // Always send a response
            console.log((result));
        } catch (err) {
            res.status(500).json({ error: "Failed to parse Python output" });
        }
    });
}); 




app.listen(prot, () => {
    console.log(`Server is running on port http://localhost:${prot}`);
});



    