// app.js

const API_URL = 'https://yasliu-lumina-engine.hf.space/analyze-shade';

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    // Tabs & Panes
    const tabUpload  = document.getElementById('tab-upload');
    const tabCamera  = document.getElementById('tab-camera');
    const paneUpload = document.getElementById('pane-upload');
    const paneCamera = document.getElementById('pane-camera');

    // Upload Elements
    const dropzone        = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzone-content');
    const fileInput       = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview    = document.getElementById('image-preview');
    const removeBtn       = document.getElementById('remove-btn');

    // Camera Elements
    const webcamVideo  = document.getElementById('webcam-video');
    const captureBtn   = document.getElementById('capture-btn');
    const cameraCanvas = document.getElementById('camera-canvas');
    let videoStream = null;

    // Flow & Results
    const analyzeBtn      = document.getElementById('analyze-btn');
    const statusMsg       = document.getElementById('status-msg');
    const statusText      = document.getElementById('status-text');
    const matchesGrid     = document.getElementById('matches-grid');
    const resetBtn        = document.getElementById('reset-btn');
    const appCard         = document.getElementById('app-card');
    const resultsContainer = document.getElementById('results-container');

    // Well-Lit Modal Elements
    const helpIcon     = document.getElementById('help-icon');
    const wellLitModal = document.getElementById('well-lit-modal');
    const modalClose   = document.getElementById('modal-close');

    // State
    let selectedFile = null;

    // --- Well-Lit Modal Logic ---
    helpIcon.addEventListener('click', () => wellLitModal.classList.add('is-open'));
    modalClose.addEventListener('click', () => wellLitModal.classList.remove('is-open'));
    wellLitModal.addEventListener('click', (e) => {
        if (e.target === wellLitModal) wellLitModal.classList.remove('is-open');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && wellLitModal.classList.contains('is-open')) {
            wellLitModal.classList.remove('is-open');
        }
    });

    // --- Tab Switching Logic ---
    tabUpload.addEventListener('click', () => switchTab('upload'));
    tabCamera.addEventListener('click', () => switchTab('camera'));

    function switchTab(tab) {
        if (tab === 'upload') {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            paneUpload.style.display = 'block';
            paneCamera.style.display = 'none';
            stopCameraStream();
            analyzeBtn.disabled = !selectedFile;
        } else {
            tabCamera.classList.add('active');
            tabUpload.classList.remove('active');
            paneCamera.style.display = 'block';
            paneUpload.style.display = 'none';
            startCameraStream();
            selectedFile = null;
            analyzeBtn.disabled = true;
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

        const reader = new FileReader();
        reader.onload = (e) => {
            // Show the preview immediately from the original file
            imagePreview.src = e.target.result;
            dropzoneContent.style.display = 'none';
            previewContainer.style.display = 'flex';

            // Re-encode through a canvas to produce a clean 3-channel JPEG.
            // This strips alpha channels, EXIF rotation, and exotic sub-formats
            // that can make cv2.imdecode crash on the backend (causing HTTP 500).
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width  = img.naturalWidth;
                canvas.height = img.naturalHeight;
                canvas.getContext('2d').drawImage(img, 0, 0);
                canvas.toBlob((blob) => {
                    selectedFile = new File([blob], 'upload.jpg', { type: 'image/jpeg' });
                    console.log('[Lumina] Re-encoded file:', selectedFile.size, 'bytes');
                    analyzeBtn.disabled = false;
                }, 'image/jpeg', 0.92);
            };
            img.src = e.target.result;
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

    // Capture-btn: snap frame from video and auto-process
    captureBtn.addEventListener('click', () => {
        if (!videoStream) return;

        cameraCanvas.width  = webcamVideo.videoWidth  || 640;
        cameraCanvas.height = webcamVideo.videoHeight || 480;

        const context = cameraCanvas.getContext('2d');
        // Un-mirror the frame: the video is CSS scaleX(-1), so we flip the canvas
        // draw to send a correctly-oriented image to the face detection backend.
        context.save();
        context.scale(-1, 1);
        context.drawImage(webcamVideo, -cameraCanvas.width, 0, cameraCanvas.width, cameraCanvas.height);
        context.restore();

        cameraCanvas.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], "selfie-capture.jpg", { type: "image/jpeg" });
                selectedFile = file;
                stopCameraStream();
                analyzeBtn.disabled = false;
                processImage(); // Auto-fire for seamless UX
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
        analyzeBtn.disabled = true;
        statusMsg.style.display = 'flex';
        statusText.innerText = 'Analyzing pigmentation...';
        statusText.style.color = 'var(--text-muted)';
        matchesGrid.innerHTML = '';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            console.log('[Lumina] Sending request to:', API_URL);
            console.log('[Lumina] File:', selectedFile.name, selectedFile.type, selectedFile.size, 'bytes');

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            // Always read the raw body text first so we can log it on any error
            const rawText = await response.text();
            console.log(`[Lumina] Response status: ${response.status}`);
            console.log('[Lumina] Raw response body:', rawText);

            if (!response.ok) {
                if (response.status === 503) {
                    statusText.innerText = 'Server is waking up — please wait a moment and try again.';
                    statusText.style.color = '#BE4B5E';
                    setTimeout(() => { resetUI(true); }, 4000);
                    return;
                }
                throw new Error(`HTTP ${response.status}: ${rawText}`);
            }

            // Parse the already-read text as JSON
            let data;
            try {
                data = JSON.parse(rawText);
            } catch (parseErr) {
                throw new Error(`JSON parse failed: ${rawText}`);
            }

            console.log('[Lumina] Parsed data:', data);

            // Backend returns null when no face is detected in the image
            if (data === null) {
                throw new Error('NO_FACE');
            }

            if (!Array.isArray(data) || data.length === 0) {
                throw new Error(`Unexpected format — got: ${JSON.stringify(data)}`);
            }

            renderMatches(data);

        } catch (error) {
            console.error('[Lumina] processImage error:', error.message);
            if (error.message === 'NO_FACE') {
                statusText.innerText = 'No face detected. Please use a clearer, well-lit selfie.';
            } else {
                statusText.innerText = 'Error analyzing image. Check the browser console (F12) for details.';
            }
            statusText.style.color = '#BE4B5E';
            setTimeout(() => { resetUI(true); }, 4000);
        }
    }

    function renderMatches(matchesArray) {
        statusMsg.style.display = 'none';

        // Switch from card view to results view
        appCard.style.display = 'none';
        resultsContainer.style.display = 'flex';

        matchesArray.forEach((match) => {
            const hexFixed = match.hex
                ? (match.hex.startsWith('#') ? match.hex : `#${match.hex}`)
                : '#C8A882';

            const cardHtml = `
                <div class="pill-card">
                    <div class="pill-swatch" style="background-color: ${hexFixed};"></div>
                    <div class="pill-info">
                        <span class="pill-brand">${escapeHtml(match.brand || 'Brand')}</span>
                        <span class="pill-product">${escapeHtml(match.product || 'Foundation')}</span>
                        <span class="pill-shade">${escapeHtml(match.name || 'Shade')}</span>
                    </div>
                </div>
            `;
            const template = document.createElement('template');
            template.innerHTML = cardHtml.trim();
            matchesGrid.appendChild(template.content.firstChild);
        });

        if (window.innerWidth <= 480) {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    // Reset whole UI back to input card
    resetBtn.addEventListener('click', () => {
        resetUI(false);
        resetUploadState();
        if (tabCamera.classList.contains('active')) {
            startCameraStream();
            analyzeBtn.disabled = true;
        }
    });

    function resetUI(isError) {
        statusMsg.style.display = 'none';
        statusText.innerText = "Analyzing pigmentation...";
        statusText.style.color = "var(--text-muted)";

        matchesGrid.innerHTML = '';
        resultsContainer.style.display = 'none';
        appCard.style.display = 'block';
        analyzeBtn.disabled = !selectedFile;

        if (window.innerWidth <= 480) {
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
