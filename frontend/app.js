// app.js

const API_URL = 'http://127.0.0.1:8000/analyze-shade';

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    // Tabs & Panes
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const paneUpload = document.getElementById('pane-upload');
    const paneCamera = document.getElementById('pane-camera');
    
    // Upload Elements
    const dropzone = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzone-content');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    
    // Camera Elements
    const webcamVideo = document.getElementById('webcam-video');
    const snapBtn = document.getElementById('snap-btn');
    const cameraCanvas = document.getElementById('camera-canvas');
    let videoStream = null;

    // Flow & Results
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusMsg = document.getElementById('status-msg');
    const statusText = document.getElementById('status-text');
    const emptyState = document.getElementById('empty-state');
    const matchesGrid = document.getElementById('matches-grid');
    const resetBtn = document.getElementById('reset-btn');
    const interactionZone = document.querySelector('.interaction-zone');

    // State
    let selectedFile = null;

    // --- Tab Switching Logic ---
    tabUpload.addEventListener('click', () => {
        switchTab('upload');
    });

    tabCamera.addEventListener('click', () => {
        switchTab('camera');
    });

    function switchTab(tab) {
        if (tab === 'upload') {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            paneUpload.style.display = 'block';
            paneCamera.style.display = 'none';
            stopCameraStream();
            analyzeBtn.innerText = "Find My Match";
            // Check if file is already there
            analyzeBtn.disabled = !selectedFile;
        } else {
            tabCamera.classList.add('active');
            tabUpload.classList.remove('active');
            paneCamera.style.display = 'block';
            paneUpload.style.display = 'none';
            startCameraStream();
            
            // If we have snapped a photo from camera, it acts exactly like an uploaded file
            // Let's reset purely for the camera view if needed
            selectedFile = null;
            analyzeBtn.disabled = true;
            analyzeBtn.innerText = "Snap & Find Match";
        }
    }

    // --- File Upload Logic ---
    
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#remove-btn')) return;
        fileInput.click();
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('drag-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('drag-active'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) handleFile(dt.files[0]);
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files.length > 0) handleFile(this.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return alert("Please upload a valid image file.");
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneContent.style.display = 'none';
            previewContainer.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadState();
    });

    function resetUploadState() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropzoneContent.style.display = 'flex';
        analyzeBtn.disabled = true;
    }


    // --- Camera Logic ---
    
    async function startCameraStream() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
                webcamVideo.srcObject = videoStream;
            } catch (err) {
                console.error("Error accessing webcam: ", err);
                alert("Could not access camera. Please allow permissions or use 'Upload Photo'.");
                switchTab('upload');
            }
        } else {
            alert("Camera not supported on your device/browser.");
            switchTab('upload');
        }
    }

    function stopCameraStream() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            webcamVideo.srcObject = null;
        }
    }

    snapBtn.addEventListener('click', () => {
        if (!videoStream) return;

        // Ensure canvas matches video resolution
        cameraCanvas.width = webcamVideo.videoWidth || 640;
        cameraCanvas.height = webcamVideo.videoHeight || 480;
        
        const context = cameraCanvas.getContext('2d');
        // If the video is mirrored via CSS, we actually capture the regular frame. 
        // For accurate matches it shouldn't matter as long as lighting is good.
        context.drawImage(webcamVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        // Convert canvas image to Blob, then File
        cameraCanvas.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], "selfie-capture.jpg", { type: "image/jpeg" });
                selectedFile = file;
                
                // Immediately stop stream and proceed to Analysis layer
                stopCameraStream();
                analyzeBtn.disabled = false;
                processImage(); // Auto-fire analyze process for magical UX
            }
        }, 'image/jpeg', 0.9);
    });

    // Clean up stream if page unloads
    window.addEventListener('beforeunload', stopCameraStream);


    // --- API & Analysis Flow ---

    analyzeBtn.addEventListener('click', () => {
        if (selectedFile) processImage();
    });

    async function processImage() {
        // UI Transition to Loading
        analyzeBtn.disabled = true;
        interactionZone.style.opacity = '0.4';
        interactionZone.style.pointerEvents = 'none';
        statusMsg.style.display = 'flex';
        emptyState.style.display = 'none';
        matchesGrid.innerHTML = '';
        resetBtn.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            
            if (!Array.isArray(data) || data.length === 0) {
                throw new Error("Invalid response format.");
            }

            renderMatches(data);

        } catch (error) {
            console.error('API Error:', error);
            statusText.innerText = "Error analyzing frame. Please try again.";
            statusText.style.color = "#FF4B4B";
            setTimeout(() => { resetUI(true); }, 3000);
        }
    }

    function renderMatches(matchesArray) {
        statusMsg.style.display = 'none';
        interactionZone.style.display = 'none'; // Fade out left controls entirely for full result focus
        
        matchesArray.forEach((match, index) => {
            const hexFixed = match.hex.startsWith('#') ? match.hex : `#${match.hex}`;
            
            const cardHtml = `
                <div class="match-card">
                    <div class="swatch-lg" style="background-color: ${hexFixed};"></div>
                    <div class="product-info">
                        <span class="match-brand">${escapeHtml(match.brand || "Brand")}</span>
                        <h4 class="match-product">${escapeHtml(match.product || "Foundation")}</h4>
                        <span class="match-shade"><strong>${escapeHtml(match.name || "Shade")}</strong></span>
                    </div>
                </div>
            `;
            const template = document.createElement('template');
            template.innerHTML = cardHtml.trim();
            matchesGrid.appendChild(template.content.firstChild);
        });

        // Show start over button below grid
        resetBtn.style.display = 'block';

        // Scroll to results seamlessly on mobile
        if (window.innerWidth <= 768) {
            document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
        }
    }

    // Reset whole UI
    resetBtn.addEventListener('click', () => {
        resetUI(false);
        resetUploadState();
        if(tabCamera.classList.contains('active')) {
            startCameraStream();
            analyzeBtn.disabled = true;
        }
    });

    function resetUI(isError) {
        interactionZone.style.opacity = '1';
        interactionZone.style.pointerEvents = 'auto';
        interactionZone.style.display = 'flex';
        statusMsg.style.display = 'none';
        statusText.innerText = "Analyzing pigmentation...";
        statusText.style.color = "var(--ink-gray)";
        
        matchesGrid.innerHTML = '';
        resetBtn.style.display = 'none';
        
        if (!isError) emptyState.style.display = 'flex';
        analyzeBtn.disabled = !selectedFile;
        
        if (window.innerWidth <= 768) {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    function escapeHtml(unsafe) {
        return (unsafe || "").toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
