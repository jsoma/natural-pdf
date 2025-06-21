(function () {
    console.log("DEBUG: app.js IIFE started");
    // Get hooks directly from the React global
    const { useState, useCallback, useMemo, useRef, useEffect, createContext, useContext } = React;
    // No need for htmPreact or manual binding with JSX

    const JSZip = window.JSZip;
    const saveAs = window.saveAs;

    // Create context for the dictionary
    const DictionaryContext = createContext();

    // Provider component to load and supply dictionary
    const DictionaryProvider = ({ children }) => {
        const [dictionary, setDictionary] = useState(new Set());

        useEffect(() => {
            async function loadWords() {
                const file = await fetch('words.txt').then(res => res.blob());
                const wordsSet = new Set((await file.text()).split(/\s+/));
                setDictionary(wordsSet);
            }
            
            loadWords();
        }, []);

        return (
            <DictionaryContext.Provider value={dictionary}>
                {children}
            </DictionaryContext.Provider>
        );
    };

    // Hook to use the dictionary in components
    const useDictionary = () => {
        return useContext(DictionaryContext);
    };

    function FlaggedText({ word }) {
        const dictionary = useDictionary();
        if(!dictionary.size) return word;

        function levenshteinDistance(a, b) {
            if (a.length === 0) return b.length;
            if (b.length === 0) return a.length;
    
            const matrix = [];
    
            for (let i = 0; i <= b.length; i++) {
                matrix[i] = [i];
            }
    
            for (let j = 1; j <= a.length; j++) {
                matrix[0][j] = j;
            }
    
            for (let i = 1; i <= b.length; i++) {
                for (let j = 1; j <= a.length; j++) {
                    const cost = (b[i - 1] === a[j - 1]) ? 0 : 1;
    
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + cost, // substitution
                        matrix[i][j - 1] + 1,        // insertion
                        matrix[i - 1][j] + 1         // deletion
                    );
                }
            }
    
            return matrix[b.length][a.length];
        }
    
        function suggestClosestMatch(word) {
            let bestMatch = null;
            let minDistance = Infinity;
    
            for (const candidate of dictionary) {
                if(Math.abs(word.length - candidate.length) > 2)
                    continue;
                const distance = levenshteinDistance(word, candidate);
    
                if (distance < minDistance) {
                    bestMatch = candidate;
                    minDistance = distance;
                }
            }
    
            return bestMatch || null; // Return the closest match or null if none found
        }
    

        const closestWord = suggestClosestMatch(word);

        return <span>
            <span style={{ backgroundColor: 'yellow' }}>{word} </span>
            <span style={{ backgroundColor: 'lightgreen' }}>{closestWord}</span>
            </span>;
    }

    function CheckedText({ inputText }) {
        const dictionary = useDictionary();

        if (!dictionary.size) return <p>Loading dictionary...</p>;

        const processText = () => {
            const wordsAndPunctuations = inputText.split(/(\b|\W)/g);

            return wordsAndPunctuations.map((word, index) => {
                if (!word.trim()) return word;

                if (
                    /[^\d]/.test(word)
                    && word.length > 2
                    && /^\w+$/.test(word)
                    && !dictionary.has(word)
                    && !dictionary.has(word.toLowerCase())
                    && !(word.endsWith('s') && dictionary.has(word.slice(0, -1).toLowerCase()))
                ) {
                    return word
                    // return <FlaggedText word={word} />;
                }

                return word;
            });
        };

        return (
            <div>
                {processText()}
            </div>
        );
    }

    // --- Region Row Component ---
    function RegionRow({ region, imageUrl, pageData, pageIndex, regionIndex, onTextChange, onEnterPress }) {
        const textRef = useRef(null);
        const canvasRef = useRef(null); 

        function handleContentEditableChange(event) {
            const newText = event.target.innerText;
            onTextChange(pageIndex, regionIndex, newText);
        }

        // Ensure the defensive check is present
        if (!region || !region.bbox || region.bbox.length !== 4) {
            console.warn("RegionRow received invalid region prop - skipping render.", { region });
            return null; // Don't render anything if region is invalid
        }

        // --- Calculate dimensions ---
        const imgScale = 1.0; // Set to 1.0 assuming bbox coords match image pixels
        const [x0, y0, x1, y1] = region.bbox;
        // Source coordinates and dimensions on the original image (now directly from bbox)
        const sourceX = x0 * imgScale; // Now just x0
        const sourceY = y0 * imgScale; // Now just y0
        const sourceWidth = (x1 - x0) * imgScale; // Now just width from bbox
        const sourceHeight = (y1 - y0) * imgScale; // Now just height from bbox

        // --- useEffect for drawing on canvas ---
        useEffect(() => {
            if (!imageUrl || !canvasRef.current) {
                return; // Don't draw if no image URL or canvas isn't ready
            }

            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // Set canvas intrinsic size to match the source snippet dimensions
                canvas.width = sourceWidth;
                canvas.height = sourceHeight;

                // Draw the specific region from the loaded image onto the canvas
                ctx.drawImage(
                    img,        // Source image
                    sourceX,    // Source X
                    sourceY,    // Source Y
                    sourceWidth,// Source Width
                    sourceHeight,// Source Height
                    0,          // Destination X (on canvas)
                    0,          // Destination Y (on canvas)
                    sourceWidth,// Destination Width (on canvas)
                    sourceHeight // Destination Height (on canvas)
                );
            };

            img.onerror = (err) => {
                console.error("Failed to load image for canvas:", imageUrl, err);
                // Optionally draw an error message or clear the canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "red";
                ctx.fillText("Error loading image", 10, 20);
            };

            img.src = imageUrl; // Start loading the image

            // No cleanup needed for image loading, but good practice for other effects
            // return () => { /* cleanup code */ };

        }, [imageUrl, region, sourceX, sourceY, sourceWidth, sourceHeight]); // Re-run effect if image or region changes


        // Style for the canvas container/scaling
        const canvasStyle = {
            // Display at 2x size (defined by sourceWidth)
            width: `${sourceWidth}px`,
            // Let height adjust automatically based on width and aspect ratio
            height: `auto`,
            // But constrain width to container
            maxWidth: '100%',
            // Remove transform scaling
            // transform: displayScale < 1 ? `scale(${displayScale})` : 'none',
            // transformOrigin: 'top left',
            // Keep other relevant styles
            borderRadius: '3px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
            display: 'inline-block', // Or 'block' depending on desired layout flow
            margin: '0 auto',
        };
        // --- END Canvas Logic ---

        const confidenceLevel = region.confidence >= 0.8 ? 'high' : (region.confidence >= 0.5 ? 'medium' : 'low');
        const confidenceText = region.confidence !== null && region.confidence !== undefined ? region.confidence.toFixed(2) : 'N/A';
        const handleInputChange = (e) => {
            onTextChange(pageIndex, regionIndex, e.target.value);
        };
        // const handleFocus = (e) => {
        //     e.target.select();
        // };
        const handleKeyDown = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                onEnterPress(e.target);
            }
        };

        // Convert to JSX
        return (
            <div className="region-item" data-region-id={region.id} data-confidence={region.confidence} data-modified={region.modified}>
                <div className="confidence-cell" data-level={confidenceLevel}>
                    {confidenceText}
                </div>

                <div className="region-content-cell">
                    {/* Replace div with canvas */}
                    <canvas
                        ref={canvasRef}
                        style={canvasStyle}
                        // Set initial width/height perhaps? Or rely purely on useEffect
                        // width={sourceWidth} // Setting via useEffect is generally better
                        // height={sourceHeight}
                        className="image-clip-canvas" // Add a class for potential styling
                    ></canvas>
                    <div
                        ref={textRef}
                        contentEditable
                        // rows={Math.max(1, Math.ceil((region.corrected_text || '').length / 50))}
                        data-page-index={pageIndex}
                        data-region-index={regionIndex}
                        data-original-text={region.ocr_text}
                        // onInput={handleInputChange}
                        // onFocus={handleFocus}
                        // onKeyDown={handleKeyDown}
                        className={`editing-content ${region.modified ? 'modified' : ''}`}
                        // onInput={handleContentEditableChange} // Handle text changes
                    >
                        <CheckedText inputText={region.corrected_text} />
                        {/* <CheckedText inputText={region.ocr_text} /> */}
                    </div>
                </div>
            </div>
        );
    }

    // --- Region Table Component ---
    function RegionTable({ regions, imageUrl, pageData, pageIndex, onTextChange, onEnterPress }) {
        // Ensure filtering is done using useMemo for efficiency
        const validRegions = useMemo(() =>
            (regions || []).filter(r => r && r.id && r.bbox && typeof r.ocr_text === 'string'),
            [regions] // Recalculate only when regions array changes
        );

        if (!validRegions || validRegions.length === 0) {
            // Convert to JSX
            return <p>No valid OCR regions to display for this page section.</p>;
        }

        // Convert to JSX
        return (
            <div className="region-list">
                {validRegions.map((region, rIndex) => (
                    <RegionRow
                        key={region.id}
                        region={region}
                        imageUrl={imageUrl}
                        pageData={pageData}
                        pageIndex={pageIndex}
                        regionIndex={rIndex}
                        onTextChange={onTextChange}
                        onEnterPress={onEnterPress}
                    />
                ))}
            </div>
        );
    }

    // --- Main Application Component ---
    function App() {
        // State for the application
        const [taskData, setTaskData] = useState(null); // Holds the parsed manifest.json
        const [imageData, setImageData] = useState({}); // Holds { relativePath: objectURL }
        const [isLoading, setIsLoading] = useState(false);
        const [error, setError] = useState(null);
        const [currentFilename, setCurrentFilename] = useState(''); // Name of the loaded zip

        // --- File Handling ---
        const handleFileChange = useCallback(async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            setIsLoading(true);
            setError(null);
            setTaskData(null);
            setCurrentFilename(file.name);
            // Revoke previous object URLs
            Object.values(imageData).forEach(URL.revokeObjectURL);
            setImageData({});

            try {
                const zip = await JSZip.loadAsync(file); // Load zip file

                // 1. Load Manifest
                const manifestFile = zip.file("manifest.json");
                if (!manifestFile) {
                    throw new Error("manifest.json not found in the zip file.");
                }
                const manifestContent = await manifestFile.async("string");
                const parsedManifest = JSON.parse(manifestContent);
                // TODO: Add validation for manifest structure?
                setTaskData(parsedManifest);

                // 2. Load Images and create Object URLs
                const imagePromises = [];
                const newImageData = {};
                zip.folder("images").forEach((relativePath, fileEntry) => {
                    if (!fileEntry.dir) {
                        imagePromises.push(
                            fileEntry.async("blob").then(blob => {
                                const objectURL = URL.createObjectURL(blob);
                                // Store URL mapped to the relative path used in manifest
                                newImageData[`images/${relativePath}`] = objectURL;
                            })
                        );
                    }
                });

                await Promise.all(imagePromises);
                setImageData(newImageData);
                console.log("Loaded images:", Object.keys(newImageData));


            } catch (err) {
                console.error("Error loading task package:", err);
                setError(`Error loading task package: ${err.message}`);
                setTaskData(null); // Clear data on error
                setImageData({});
                setCurrentFilename('');
            } finally {
                setIsLoading(false);
                // Reset file input value so the same file can be loaded again
                event.target.value = null;
            }
        }, [imageData]); // Depend on imageData to revoke old URLs

        // --- Text Area Change Handler ---
        const handleTextChange = (pageIndex, regionIndex, newText) => {
            setTaskData(prevData => {
                if (!prevData) return null; // Handle null state
        
                const newData = { ...prevData }; // Shallow copy top-level object
                const newPages = [...newData.pages]; // Create a shallow copy of pages array
        
                // Ensure pageIndex and regionIndex are within bounds to prevent errors
                if (newPages[pageIndex] && Array.isArray(newPages[pageIndex].regions)) {
                    const newRegions = [...newPages[pageIndex].regions]; // Shallow copy regions array
                    const region = newRegions[regionIndex];
        
                    // Update only the specific region's corrected_text and modified fields
                    if (region) {
                        region.corrected_text = newText;
                        region.modified = newText !== region.ocr_text;
                        newRegions[regionIndex] = region; // Reassign updated region to array
        
                        newPages[pageIndex].regions = newRegions; // Reassign updated regions array to page
                    }
                }
        
                newData.pages = newPages; // Reassign updated pages array to data object
                return newData;
            });
        };

        // --- Enter Key Navigation Handler ---
        const handleEnterNavigation = (currentTextArea) => {
            const allTextAreas = Array.from(document.querySelectorAll('.text-content-input'));
            const currentIndex = allTextAreas.indexOf(currentTextArea);

            if (currentIndex > -1 && currentIndex < allTextAreas.length - 1) {
                const nextTextArea = allTextAreas[currentIndex + 1];
                nextTextArea.focus();
            }
        };

        // --- UI Rendering (Convert to JSX) ---
        return (
            <DictionaryProvider>
            <div className="app-container">
                <div className="task-loader">
                    <label htmlFor="zip-input">Load Correction Task Package (.zip): </label>
                    <input type="file" id="zip-input" accept=".zip" onChange={handleFileChange} disabled={isLoading} />
                    {isLoading && <span> Loading...</span>}
                </div>

                {error && <div className="error-message" style={{ color: 'red', margin: '10px', padding: '10px', border: '1px solid red' }}>{error}</div>}

                {!isLoading && !taskData && !error && (
                    <div className="initial-message">
                        <p>Please load a .zip task package to begin.</p>
                    </div>
                )}

                {taskData && (
                    <div className="task-content">
                        <h2>Task: {currentFilename}</h2>
                        <p>PDF Source: {taskData.pdfs && taskData.pdfs.length > 0 ? taskData.pdfs[0].source : (taskData.pages[0]?.pdf_source || 'Unknown')} ({taskData.pages?.length || 0} pages)</p>

                        <div className="controls-container">
                            {/* TODO: Add Export Functionality */}
                            <button id="export-corrections" className="export-btn">Export Corrections JSON</button>
                        </div>

                        <div className="pages-container">
                           {taskData.pages.map((page, pIndex) => (
                               <div className="page-section" key={page.image_path}>
                                   <div className="page-title">
                                       <h3>Page {page.page_number} (Source: {page.pdf_short_id})</h3>
                                   </div>
                                   <RegionTable
                                        regions={page.regions}
                                        imageUrl={imageData[page.image_path]} /* Pass the object URL */
                                        pageData={page} /* Pass page metadata (width, height) */
                                        pageIndex={pIndex}
                                        onTextChange={handleTextChange}
                                        onEnterPress={handleEnterNavigation}
                                   />
                               </div>
                           ))}
                        </div>
                    </div>
                )}
            </div>
            </DictionaryProvider>
        );
    }

    console.log("DEBUG: Mounting React app...");
    // Mount the app to the DOM using ReactDOM.createRoot
    const root = ReactDOM.createRoot(document.getElementById('app'));
    root.render(<App />);
    console.log("DEBUG: React app mount initiated.");

})(); // Immediately invoke the function 
