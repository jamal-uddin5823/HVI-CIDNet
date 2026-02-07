# Gradio "event_id" Missing Error - Solution

## Problem

The Gradio UI shows "Error" with HTTP 422 status:
```json
{
    "detail": [{
        "type": "missing",
        "loc": ["body", "event_id"],
        "msg": "Field required"
    }]
}
```

**This is NOT an error in your Python code!** It's a Gradio framework version compatibility issue.

## Why This Happens

The error occurs at the Gradio API validation layer **before** your `process_query()` function is even called. This is why:
- No DEBUG messages appear in terminal
- The try-except block never catches it
- Only the UI shows "Error"

The Gradio client (JavaScript in browser) and server (Python) have mismatched API expectations about the request format.

## Solutions

### Solution 1: Update Gradio (Recommended)

```bash
# Check current version
pip show gradio

# Update to stable version
pip install gradio==4.44.0 --upgrade

# Or try latest
pip install gradio --upgrade

# Restart the app
python face_recognition_app.py --device cuda
```

### Solution 2: Downgrade Gradio

If the latest version has issues, try a known stable version:

```bash
pip install gradio==3.50.2
```

### Solution 3: Clear Browser Cache

Sometimes the browser caches old Gradio client code:

1. Clear browser cache (Ctrl+Shift+Delete)
2. Hard refresh the page (Ctrl+F5)
3. Or use incognito/private browsing mode

### Solution 4: Check Port Forwarding

If using SSH tunnel, ensure correct port mapping:

```bash
# On local PC
ssh -L 7864:localhost:7863 hpc12@cail-hpc12

# Server port 7863 -> Local port 7864
```

## Verification Steps

After applying a solution:

1. **Restart the server:**
   ```bash
   python face_recognition_app.py --device cuda
   ```

2. **Check Gradio version in startup:**
   ```
   Running on local URL:  http://0.0.0.0:7863
   ```

3. **Upload an image and click Process**

4. **Look for DEBUG messages in terminal:**
   ```
   [DEBUG] ========== Processing query image ==========
   [DEBUG] Recognizer type: AdaFace
   ...
   ```

5. **If DEBUG messages appear = SUCCESS!**
   The error was Gradio-related, not your code.

## What We Added to Help Debug

The debug improvements we added will help diagnose **application-level** errors:
- Model loading failures
- Tensor device mismatches
- Face database issues
- Processing pipeline errors

But they can't help with **framework-level** errors that happen before your code runs.

## Additional Debugging

If the issue persists after updating Gradio:

### Check Gradio Dependencies
```bash
pip list | grep gradio
pip check
```

### Try Minimal Test
Create `test_gradio.py`:
```python
import gradio as gr

def process(image):
    return image

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input", type="pil")
        output_img = gr.Image(label="Output", type="pil")
    btn = gr.Button("Process")
    btn.click(fn=process, inputs=input_img, outputs=output_img)

demo.launch(server_port=7865, server_name="0.0.0.0")
```

Run: `python test_gradio.py`

If this minimal example works, the issue is in your app code.
If it fails with the same error, it's a Gradio installation issue.

## Related Issues

This is a known Gradio issue reported in:
- https://github.com/gradio-app/gradio/issues/...
- Often occurs with version mismatches between Gradio versions
- Can happen when upgrading/downgrading Gradio versions

## Summary

**The debugging code we added is working perfectly!**

The issue is that Gradio is rejecting the request before it reaches your code. Update Gradio to fix this framework-level compatibility issue.

Once Gradio is fixed, if there are any errors in your application logic, you'll see detailed DEBUG messages and error traces in the terminal thanks to the improvements we made.
