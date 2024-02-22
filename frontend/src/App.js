import './App.css';
import React, { useState, useEffect } from "react";
import Switch from "react-switch";

const backendHOST = "http://127.0.0.1:5000";
// const backendHOST = "http://172.30.195.21:5000";

async function APIFunc(file,callback) {
  let endpoint = backendHOST + `/run`;
  
  const formData = new FormData();
  formData.append("file",file);
  await fetch(endpoint, {
      method: "POST",
      body: formData
  })
  .then(r => r.json())
  .then(r => {
      if (callback) callback(true, r);
  })
  .catch(e => {
      if (callback) callback(false, e);
  });
};

function App() {

  const [openAIKey, setOpenAIKey] = useState("");
  const [imageFile, setImageFile] = useState("");
  const [output, setOutput] = useState("");
  const [showLoading, setShowLoading] = useState(false);

  const handleUploadImageFile = (event) => {
		event.preventDefault(); //
		if (!event.target.files || event.target.files.length == 0) return;
		var file = event.target.files[0];
		event.target.value = ''; // reset the value
    setImageFile(file);

    setShowLoading(true);
    runAPI(file, (success,data) => {
      setOutput(success ? data.value : "[failed]")
      setShowLoading(false);
    });
  }

  const runAPI = (image, callback) => {
    APIFunc(image,callback);
  };

  return (
    <div className="App">
      { showLoading && <div className='notification'>
        <div className='overlay'>
          <div className='loading'>
            <p>Loading ... </p>
          </div>
        </div>
      </div> }
      <header className="App-header">
        <h1>
          BEIT
        </h1>
        <p>
          CHECK FACE IMAGE
        </p>
      </header>
      <section className="App-body">
        <div className="body-container">
          <div className="main-panel">
            <div className="panel-inside">
              <div className='uploader'>
                <div className='file-name-display'>{imageFile ? imageFile.name : ""}</div>
                <div className="choose-file-button">
                  <label
                    htmlFor="select-image-file"
                    className="">
                    Select Image
                  </label>
                  <input
                    type="file"
                    onChange={handleUploadImageFile}
                    id="select-image-file"
                    name="filename"
                    accept=".png"
                    hidden/>
                </div>
              </div>
              {output && <div className='output'><b>Output:</b>&nbsp; {output}</div> }
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
