# Fix: Model Re-upload Issue

## Problem Description

After uploading a model successfully and loading it in the web app, users were unable to upload a different model. The "Upload Another Model" button would reset the UI state, but the file input element would not trigger the `onChange` event when selecting a new file.

## Root Cause

HTML file input elements do not trigger the `onChange` event when:
1. The user clicks the input again after selecting a file
2. The selected file has the same name as the previously selected file
3. The input's value hasn't been explicitly reset

This is a browser behavior to prevent duplicate file selections.

## Solution

The fix involves:
1. Adding a React `useRef` hook to reference the file input element
2. Resetting the file input's value to an empty string when the "Upload Another Model" button is clicked

### Code Changes

**File:** `frontend/src/components/ModelDropzone.js`

```javascript
// Added useRef import
import React, { useState, useCallback, useEffect, useRef } from 'react';

// Created ref for file input
const fileInputRef = useRef(null);

// Attached ref to input element
<input
  type="file"
  id="model-upload"
  ref={fileInputRef}
  accept=".pth,.pt"
  onChange={handleChange}
  className="hidden"
/>

// Reset file input value when uploading another model
onClick={() => {
  setFileName(null);
  setPendingFile(null);
  onAnalysisComplete(null);
  setError(null);
  // Reset the file input element to allow selecting the same file again
  if (fileInputRef.current) {
    fileInputRef.current.value = '';
  }
}}
```

## Testing Instructions

### Manual Testing

1. Start the backend server:
   ```bash
   cd ViXNet/web_app/backend
   python app.py
   ```

2. Start the frontend development server:
   ```bash
   cd ViXNet/web_app/frontend
   npm install
   npm start
   ```

3. Test the fix:
   - Upload a model file (.pth or .pt)
   - Wait for the model to load successfully
   - Click "Upload Another Model" button
   - Try uploading a different model file
   - Verify that the upload works correctly
   - Try uploading the same model file again
   - Verify that it works (this was previously broken)

### Expected Behavior

- After clicking "Upload Another Model", users should be able to:
  - Select and upload a different model file
  - Select and upload the same model file again
  - The file input should work consistently on every upload

## Impact

This fix ensures that users can upload multiple models in sequence without refreshing the page, improving the user experience and workflow efficiency.

## Related Files

- `frontend/src/components/ModelDropzone.js` - Main component with the fix
- `backend/app.py` - Backend API that handles model uploads (unchanged)
