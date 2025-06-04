import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  // Choose to see the results or the Queree main content
  const [queree, seeQueree] = useState(true); 
  const [results, setResults] = useState(null);

  const seeResults = () => {
    seeQueree(false);
  };
  const returnToQueree = () => {
    seeQueree(true);
  };

  return (
    <div className="App">
      {queree ? (
        <PageA showResults={seeResults} sendData = {setResults}/> 
      ) : (
        <PageB returnToMain={returnToQueree} receivedData = {results}/>
      )}
    </div>
  );
}

// QUEREE MAIN CONTENT
function PageA({ showResults, sendData}) {
  // States
  const socket = useRef(null);
  const [text, setText] = useState(()=>{
    return localStorage.getItem('input') || ''
  });
  const [dir, setDir] = useState(() => {
    return localStorage.getItem('dir') || '...';
  });
  useEffect(() => {
    localStorage.setItem('dir', dir);
  }, [dir]);
  useEffect(() => {
    localStorage.setItem('input', text);
  }, [text]);

  const [response, setResponse] = useState('Engine off'); 
  const [value, setProgress] = useState(1)
  const max = 100 

  // Update progress bar number
  const progressRef = useRef(null);
  useEffect(() => {
    if (progressRef.current) {
      progressRef.current.setAttribute('data-value', value);
    }
  }, [value]);

  // Listen for the directory path sent from the main process & update dir
  // Once the user picks the directory main.js sends back the event with the path
  useEffect(() => {
    window.electron.on('selected-directory', (event, path) => {
      setDir(path);  
    });
  }, []);
  // From select directory button
  const openDirectoryDialog = () => {
    window.electron.send('open-directory-dialog');  
  };

  // Focus the input when the component mounts
  // Update the text state when input changes
  const inputRef = useRef(null);
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus(); 
    }
  }, []);
  const handleTextChange = (event) => {
    setText(event.target.value); 
  };


  // Starts web socket with backend once the user clicks send btn
  const send = async () => {
    setProgress(1)
    setResponse("Starting engine...")

    const socket = new WebSocket('ws://localhost:8000/ws');

    socket.onclose = () => {
      console.log('WebSocket connection closed');
      setResponse("Failed to connect the engine: Not ready yet, try again")
      setProgress(1)

    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setResponse("Failed to connect the engine: Not ready yet, try again")
      setProgress(1)
    };

    // Send message to verify connection
    socket.onopen = () => {
      console.log('Connected to WebSocket server');
      socket.send(JSON.stringify({ query: text, directory: dir}));
    };

    // Handle messages from the backend
    socket.onmessage = (event) => {
      try { // JSON
        const data = JSON.parse(event.data);
        console.log('Received from server (JSON):', data);
        setResponse(data.message)
        setProgress(parseInt(data.progress))
        if(parseInt(data.progress)==100){
          console.log("Completed, sending json to results page")
          sendData(data.output)
        }
      } catch (e) { // Plain texts
        console.log('Received from server (plain text):', event.data);
      }
    };
  };


  const handleMainButton = async () => {
    if (value === 1 || value === 100) {
      if (!socket.current || socket.current.readyState === WebSocket.CLOSED) {
        socket.current = new WebSocket('ws://localhost:8000/ws');
        
        socket.current.onopen = () => {
          console.log('Connected to WebSocket server');
          socket.current.send(JSON.stringify({ query: text, directory: dir }));
          setProgress(0);
          setResponse("Starting engine...");
        };
        
        socket.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('Received from server (JSON):', data);
            setResponse(data.message);
            setProgress(parseInt(data.progress));
            if (parseInt(data.progress) === 100) {
              console.log("Completed, sending json to results page");
              sendData(data.output);
              // Optionally close after completion:
              socket.current.close();
            }
          } catch (e) {
            console.log('Received from server (plain text):', event.data);
          }
        };
        
        socket.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setResponse("Failed to connect the engine: Not ready yet, try again");
          setProgress(1);
        };
        
        socket.current.onclose = () => {
          console.log('WebSocket connection closed');
          setProgress(1);
        };
      }
    } else {
      // If socket exists and open, close it
      //if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      //  socket.current.close();
      //}
      //setResponse('Connection closed due to invalid value.');
      //setProgress(1); 
    }
  };
  

  return (
    <div className="App bg-black">
      <header className="bg-black h-10 flex items-center justify-center">
        <h1 className="text-white font-bold">Q U E R E E</h1>
      </header>

      <div className="flex">
        <div className="bg-black w-20 flex flex-col">
          <div className='bg-transparent mt-3 flex-1 text-center' ><i class="fa-solid fa-lg fa-circle-user text-white"></i></div>
          <div className='bg-transparent mt-3 flex-1 text-center' ><i class="fa-solid fa-lg fa-cube text-white"></i></div>
          <div className='bg-transparent mt-3 flex-1 text-center' ><i class="fa-solid fa-lg fa-pen-to-square text-white"></i></div>
          <div className='bg-transparent mt-3 flex-1 text-center' ><i class="fa-solid fa-lg fa-gears text-white"></i></div>
        </div>

      <div
        className="bg-stone-400 w-full p-4 pl-8 pt-5 relative"
        style={{
          //backgroundImage: "url('file:///home/bran/Documents/QUEREE/desktop-app/gui/src/bg4.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundColor: "rgb(39, 39, 39)"
        }}
      >       
          {/* Directory Button */}
          <div className="mb-4">
            <input
              type="button"
              onClick={openDirectoryDialog}
              className="h-8 w-30 rounded-full bg-indigo-900 text-white text-[14px]"
              value="Select directory"
            />
            <span className="bg-transparent mx-8 text-sm text-white">{dir}</span>
          </div>

          {/* Text Area */}
          <div className="bg-transparent flex">
            <textarea
              ref={inputRef}
              value={text}
              onChange={handleTextChange}
              className="h-10 w-160 p-2 pt-2 pl-4 text-sm bg-stone-100 rounded-full"
              placeholder="Search..."
            />
            <div className="">
              <input
                type="button"
                className={`w-20 h-8 p-1 ml-6 mt-1 rounded-full text-white text-sm ${
                  value === 1 || value === 100?  'bg-green-600' : 'bg-amber-600'
                }`}
                value={value === 1 || value === 100?  "Start" : "Cancel"}
                onClick={send}
              />

            </div>
          </div>

          {/* Info Section */}
          <div className="bg-transparent text-white mt-6 ml-2 mb-1 w-180 text-sm">
            <span>{response}</span>
          </div>

          {/* Progressbar Section */}
          <div className="mb-6">
            <div className='flex'>
                <progress
                ref={progressRef}
                value={value}
                className="custom-progress"
                max={max}
              ></progress>

              <input
                type="button"
                className={`ml-6 w-20 h-6 -mt-2 rounded-full text-sm ${
                  value === 100 ? 'bg-white text-black' : 'bg-transparent text-transparent'
                }`}
                value="Show"
                onClick={showResults}
              />            
              </div>
          </div>
        </div>
      </div>
    </div>
  );
}


function PageB({ returnToMain, receivedData }) {
  // TODO: Function to handle image click
  const handleImageClick = (imagePath) => {
    console.log(`Opening file: ${imagePath}`);
  };

  return (
    <div className="p-6 bg-black min-h-screen">
      <h1 className="text-2xl font-bold text-center text-white ">R E S U L T S</h1>
      <p className="text-center text-sm text-gray-400 mb-4">Click the image to open the file</p>
      <div className="grid grid-cols-5 gap-6">
        {receivedData.map((item, index) => (
          <div key={index} className="border border-gray-300 rounded-lg shadow-md p-2 text-center bg-white hover:shadow-lg transition-shadow duration-300">
            <img
              src={item.image_path}
              onClick={() => handleImageClick(item.image_path)}
              className="w-24 h-24 cursor-pointer mx-auto rounded-md"
              alt={`Image ${index + 1}`}
            />
            <p className="mt-2 text-gray-700 text-sm">Score: {item.similarity.toFixed(4)}</p>
          </div>
        ))}
      </div>
      <button onClick={returnToMain} className="mt-14 px-6 py-3 text-sm bg-orange-500 text-white rounded-md hover:bg-blue-600 transition-colors duration-300 mx-auto block">
        Return to Queree
      </button>
    </div>
  );
}


export default App;