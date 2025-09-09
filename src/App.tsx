// src/App.jsx
import React, { useEffect } from "react";
import * as ort from "onnxruntime-web";

function App() {
  console.log({ ort });
  useEffect(() => {
    async function runInference() {
      // Path must be relative to public folder in Vite projects
      const modelUrl = `mymodel.onnx`;

      // Create session
      const session = await ort.InferenceSession.create(modelUrl);

      // Example: prepare one input (float32 tensor)
      const input = new ort.Tensor(
        "float32",
        [
          /*data*/
        ],
        [
          /*dimensions*/
        ]
      );
      const feeds = { input_name: input };

      // Run inference
      const results = await session.run(feeds);

      // Log output
      console.log(results);
    }
    runInference();
  }, []);

  return <div>ONNXRuntime Web Inference Example</div>;
}

export default App;
