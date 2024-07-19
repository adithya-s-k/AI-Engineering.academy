import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { Token } from './components/Token';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
// import faker from 'faker';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  // Allow user to set tokenizer and text via URL query parameters
  const urlParams = new URLSearchParams(window.location.search);
  const tokenizerParam = urlParams.get('tokenizer');
  const textParam = urlParams.get('text');

  const [margins] = useState([]);
  const [tokenizerData, setTokenizerData] = useState({
    'Xenova/gpt-4': {
      tokenIds: [],
      decodedTokens: [],
      margins: [],
    },
    'Xenova/llama-tokenizer': {
      tokenIds: [],
      decodedTokens: [],
      margins: [],
    },
  });
  const [outputOption, setOutputOption] = useState('text');
  const [showVisualization, setShowVisualization] = useState(true);
  const [tokenizer, setTokenizer] = useState(tokenizerParam ?? 'Xenova/gpt-4');

  const textareaRef = useRef(null);
  const outputRef = useRef(null);
  // Create a reference to the worker object.
  const worker = useRef(null);
  const options = {
    responsive: true,

    plugins: {
      legend: {
        position: 'bottom',
      },
    },
  };
  const labels = Object.keys(tokenizerData);
  const data = {
    labels: labels,
    datasets: [
      {
        label: 'Tokenizer Arena',
        data: labels.map(
          (tokenizerName) => tokenizerData[tokenizerName].tokenIds.length
        ),
        backgroundColor: 'rgba(216, 180, 254, 0.7)',
      },
    ],
  };

  // We use the `useEffect` hook to set up the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module',
      });
    }
    function tokeniserStateHandler(
      modelName,
      tokenIds,
      decodedTokens,
      margins,
      tokenizerData,
      setTokenizerData
    ) {
      setTokenizerData((prevData) => ({
        ...prevData,
        [modelName]: {
          tokenIds: [...tokenIds],
          decodedTokens: [...decodedTokens],
          margins: [...margins],
        },
      }));
    }
    const onMessageReceived = (e) => {
      tokeniserStateHandler(
        e.data.model_id,
        e.data.token_ids,
        e.data.decoded,
        e.data.margins,
        tokenizerData,
        setTokenizerData
      );
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener('message', onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () =>
      worker.current.removeEventListener('message', onMessageReceived);
  }, [tokenizerData]);

  const onInputChange = useCallback(
    (e) => {
      // const model_id = tokenizer;
      const text = e.target.value;

      if (text.length > 10000) {
        setOutputOption(null);
        console.log(
          'User most likely pasted in a large body of text (> 10k chars), so we hide the output (until specifically requested by the user).'
        );
      }
      Object.entries(tokenizerData).forEach(([model_id]) => {
        worker.current.postMessage({ model_id, text });
      });
    },
    [tokenizerData]
  );

  useEffect(() => {
    if (textParam) {
      onInputChange({ target: { value: textParam } });
    }
  }, [onInputChange, textParam]);

  const onTokenizerChange = useCallback((e) => {
    const model_id = e.target.value;
    setTokenizer(model_id);
    if (model_id !== 'custom') {
      setTokenizerData((prevData) => ({
        ...prevData,
        [model_id]: {
          tokenIds: [],
          decodedTokens: [],
          margins: [],
        },
      }));
    }
    worker.current.postMessage({ model_id, text: textareaRef.current.value });
  }, []);

  const removeTokenizer = (tokenizerName) => {
    setTokenizerData((prevData) => {
      // Create a copy of the previous state
      const updatedData = { ...prevData };
      // Remove the specified tokenizer from the state
      setTokenizer('');
      delete updatedData[tokenizerName];
      return updatedData;
    });
  };

  return (
    <div className="w-full max-w-100vw flex flex-col gap-4 items-center">
      <div>
        <h1 className="text-5xl font-bold mb-2">Tokenizer Arena</h1>
        <h2 className="text-lg font-normal">
          Easily compare between different tokenizers simultaneously
        </h2>
        <h3>
          Create by{' '}
          <a
            className="font-semibold text-white"
            href="https://twitter.com/adithya_s_k"
          >
            Adithya S K
          </a>{' '}
          on top of{' '}
          <a
            className="font-semibold text-white"
            href="https://github.com/xenova/transformers.js"
          >
            Transformer js
          </a>
        </h3>
      </div>

      <div>
        <select
          value={tokenizer}
          onChange={onTokenizerChange}
          className="border-gray-700 bg-gray-100/10  text-slate-400 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2"
        >
          <option value="Xenova/gpt-4">
            gpt-4 / gpt-3.5-turbo / text-embedding-ada-002
          </option>
          <option value="Xenova/text-davinci-003">
            text-davinci-003 / text-davinci-002
          </option>
          <option value="Xenova/gpt-3">gpt-3</option>
          <option value="Xenova/grok-1-tokenizer">Grok-1</option>
          <option value="Xenova/claude-tokenizer">Claude</option>
          <option value="Xenova/mistral-tokenizer">Mistral</option>
          <option value="Xenova/gemma-tokenizer">Gemma</option>
          <option value="Xenova/llama-tokenizer">LLaMA / Llama 2</option>
          <option value="AdithyaSK/LLama3Tokenizer">Llama 3</option>
          <option value="microsoft/Phi-3-mini-128k-instruct">Phi 3</option>
          <option value="Xenova/c4ai-command-r-v01-tokenizer">
            Cohere Command-R
          </option>
          <option value="Xenova/t5-small">T5</option>
          <option value="Xenova/bert-base-cased">bert-base-cased</option>
        </select>
      </div>

      <textarea
        ref={textareaRef}
        onChange={onInputChange}
        rows="2"
        className="font-mono text-md text-white block w-full p-2.5 bg-gray-100/10 rounded-lg border-gray-700 "
        placeholder="Enter some text"
        defaultValue={textParam ?? textareaRef.current?.value ?? ''}
      ></textarea>

      <div className="grid grid-cols-2 gap-4 w-full">
        {Object.entries(tokenizerData).map(([tokenizerName, data]) => (
          <div key={tokenizerName} className="flex flex-col">
            <p>Tokenizer Name: {tokenizerName}</p>
            <div className="flex justify-center gap-5">
              <div className="flex flex-col">
                Tokens: {data.tokenIds.length.toLocaleString()}
              </div>
              <div className="flex flex-col">
                Characters :{' '}
                {(textareaRef.current?.value.length ?? 0).toLocaleString()}
              </div>
            </div>
            <div
              ref={outputRef}
              className="font-mono text-black text-lg p-2.5  bg-gray-100/10 rounded-t-lg whitespace-pre-wrap text-left h-[150px] overflow-y-auto w-full"
            >
              {outputOption === 'text' ? (
                data.decodedTokens.map((token, index) => (
                  <Token
                    key={index}
                    text={token}
                    position={index}
                    margin={margins[index]}
                  />
                ))
              ) : outputOption === 'token_ids' ? (
                <div className="text-white">[{data.tokenIds.join(', ')}]</div>
              ) : null}
            </div>
            <button
              className="bg-slate-800 rounded-b-lg"
              onClick={() => removeTokenizer(tokenizerName)}
            >
              Remove
            </button>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 self-end">
        <div className="flex items-center">
          <input
            checked={outputOption === 'text'}
            onChange={() => setOutputOption('text')}
            id="output-radio-1"
            type="radio"
            value=""
            name="output-radio"
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
          />
          <label
            htmlFor="output-radio-1"
            className="ml-1 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            Text
          </label>
        </div>
        <div className="flex items-center">
          <input
            checked={outputOption === 'token_ids'}
            onChange={() => setOutputOption('token_ids')}
            id="output-radio-2"
            type="radio"
            value=""
            name="output-radio"
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
          />
          <label
            htmlFor="output-radio-2"
            className="ml-1 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            Token IDs
          </label>
        </div>
        <div className="flex items-center">
          <input
            checked={outputOption === null}
            onChange={() => setOutputOption(null)}
            id="output-radio-3"
            type="radio"
            value=""
            name="output-radio"
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
          />
          <label
            htmlFor="output-radio-3"
            className="ml-1 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            Hide
          </label>
        </div>
        <div>
          <input
            id="output-checkbox-4"
            type="checkbox"
            checked={showVisualization}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
            onChange={(e) => setShowVisualization(e.target.checked)}
          />
          <label
            htmlFor="output-checkbox-4"
            className="ml-1 text-sm font-medium text-gray-900 dark:text-gray-300"
          >
            Show Visualization
          </label>
        </div>
      </div>
      {showVisualization && (
        <div className="w-1/2">
          {' '}
          <Bar options={options} data={data} />{' '}
        </div>
      )}
    </div>
  );
}

export default App;
