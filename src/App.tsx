// src/App.jsx
import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

function App() {
  const [output, setOutput] = useState<any>(null);
  console.log({ ort });
  useEffect(() => {
    async function runInference() {
      // Path must be relative to public folder in Vite projects
      const modelUrl = `/mymodel.onnx`;

      // Create session
      const session = await ort.InferenceSession.create(modelUrl);

      // Example: prepare one input (float32 tensor)
      const input = new ort.Tensor("float32", [0.5], [1, 1]);
      console.log({ input });
      const feeds = { x: input };

      // Run inference
      const results = await session.run(feeds);
      setOutput(results);
      // Log output
      console.log({ results });
    }
    runInference();
  }, []);

  return (
    <div>
      ONNXRuntime Web Inference Example
      <div>
        {/* Display output as JSON */}
        <pre>
          {/* {output ? JSON.stringify(output, null, 2) : "Running inference..."} */}
          {output ? output?.linear_3?.cpuData["0"] : "Running...."}
        </pre>
      </div>
    </div>
  );
}

export default App;
