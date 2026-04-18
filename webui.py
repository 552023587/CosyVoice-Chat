import sys
sys.path.append('third_party/Matcha-TTS')
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import io
import os
import uuid
import tempfile
from typing import Optional
import numpy as np
import soundfile as sf
from openai import OpenAI

app = FastAPI(title="CosyVoice TTS API", description="CosyVoice3 文本转语音API接口")

# 全局加载模型
cosyvoice = None
MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
TEMP_DIR = tempfile.gettempdir()

class TTSRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = "You are a helpful assistant. <|endofprompt|>"
    reference_audio_path: str = "./asset/female.mp3"

class ChatRequest(BaseModel):
    message: str
    llm_url: Optional[str] = "http://127.0.0.1:8001/v1/chat/completions"
    system_prompt: Optional[str] = "你是一个友好的AI助手，请用简洁的中文回答用户的问题。"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    reference_audio_path: Optional[str] = "./asset/female.mp3"
    prompt_text: Optional[str] = "You are a helpful assistant. <|endofprompt|>"

@app.on_event("startup")
async def startup_event():
    global cosyvoice
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"Model directory {MODEL_DIR} not found")
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")

@app.get("/")
async def root():
    return {"message": "CosyVoice TTS API is running", "model_loaded": cosyvoice is not None}

@app.get("/play", summary="在线播放页面")
async def play_page():
    """返回在线播放页面"""
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>CosyVoice TTS 在线播放</title>
    <meta charset="utf-8">
    <style>
        body { max-width: 800px; margin: 50px auto; padding: 20px; font-family: Arial, sans-serif; }
        textarea { width: 100%; height: 150px; padding: 10px; margin: 10px 0; font-size: 16px; }
        button { padding: 10px 30px; font-size: 18px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .audio-container { margin: 20px 0; }
        .reference { margin: 15px 0; }
        select, input { padding: 8px; font-size: 16px; min-width: 300px; }
        .status { margin: 10px 0; color: #666; }
    </style>
</head>
<body>
    <h1>CosyVoice 文本转语音</h1>
    <div class="reference">
        <label>参考音频：</label>
        <select id="reference">
            <option value="./asset/female.mp3">female.mp3</option>
            <option value="./asset/dingzhen.mp3">dingzhen.mp3</option>
            <option value="./asset/yujie.mp3">yujie.mp3</option>
        </select>
    </div>
    <div class="reference">
        <label>Prompt Text：</label><br>
        <textarea id="prompt" style="height: 50px; margin-top: 5px;">You are a helpful assistant. &lt;|endofprompt|&gt;</textarea>
    </div>
    <div>
        <label>输入文本：</label><br>
        <textarea id="text" placeholder="输入要转换的文本...">收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。</textarea>
    </div>
    <button id="generate" onclick="generate()">生成并播放</button>
    <div class="status" id="status"></div>
    <div class="audio-container" id="audioContainer"></div>

<script>
async function generate() {
    const text = document.getElementById('text').value.trim();
    const reference = document.getElementById('reference').value;
    const prompt = document.getElementById('prompt').value;
    const button = document.getElementById('generate');
    const status = document.getElementById('status');
    const container = document.getElementById('audioContainer');

    if (!text) {
        alert('请输入文本');
        return;
    }

    button.disabled = true;
    status.textContent = '正在生成语音...';

    try {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        container.innerHTML = `
            <h3>生成完成：</h3>
            <audio controls autoplay>
                <source src="${audioUrl}" type="audio/wav">
                您的浏览器不支持音频播放
            </audio>
            <br>
            <a href="${audioUrl}" download="tts_output.wav" style="display: inline-block; margin-top: 10px;">下载音频</a>
        `;
        status.textContent = '生成完成';
    } catch (error) {
        status.textContent = '错误：' + error.message;
    } finally {
        button.disabled = false;
    }
}
</script>
</body>
</html>
'''
    return HTMLResponse(content=html)

@app.post("/tts/stream", summary="文本转语音（直接在线播放）")
async def text_to_speech_stream(request: TTSRequest):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not os.path.exists(request.reference_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {request.reference_audio_path}")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_buffer = io.BytesIO()
        full_audio = None
        for chunk in cosyvoice.inference_instruct2(
            request.text,
            request.prompt_text,
            request.reference_audio_path,
            stream=True
        ):
            speech_np = chunk['tts_speech'].squeeze(0).cpu().numpy()
            if full_audio is None:
                full_audio = speech_np
            else:
                full_audio = np.concatenate([full_audio, speech_np])

        sf.write(audio_buffer, full_audio, cosyvoice.sample_rate, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

@app.post("/tts/download", summary="文本转语音（下载）")
async def text_to_speech_download(request: TTSRequest):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not os.path.exists(request.reference_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {request.reference_audio_path}")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(TEMP_DIR, output_filename)

        for i, j in enumerate(cosyvoice.inference_instruct2(
            request.text,
            request.prompt_text,
            request.reference_audio_path,
            stream=False
        )):
            torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="cosyvoice_tts.wav",
            content_disposition=f"attachment; filename=cosyvoice_tts.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

@app.get("/chat", summary="LLM对话页面")
async def chat_page():
    """返回LLM对话页面，原生流式TTS边生成边播放"""
    sample_rate = cosyvoice.sample_rate if cosyvoice else 24000
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LLM + CosyVoice 语音对话</title>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.3/howler.min.js"></script>
    <style>
        body {{ max-width: 900px; margin: 30px auto; padding: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }}
        .container {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chat-container {{ height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; background: #fafafa; }}
        .message {{ margin-bottom: 15px; padding: 10px 15px; border-radius: 8px; max-width: 80%; }}
        .user {{ background: #007bff; color: white; margin-left: auto; }}
        .assistant {{ background: #e9ecef; color: #333; margin-right: auto; }}
        .assistant .text {{ margin-bottom: 8px; }}
        .play-btn {{ background: #28a745; color: white; border: none; padding: 5px 15px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
        .play-btn:hover {{ background: #218838; }}
        .play-btn:disabled {{ background: #ccc; cursor: not-allowed; }}
        .controls {{ display: grid; grid-template-columns: 1fr; gap: 15px; }}
        .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        @media (max-width: 600px) {{ .row {{ grid-template-columns: 1fr; }} }}
        label {{ font-weight: bold; margin-bottom: 5px; display: inline-block; }}
        input, textarea, select {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; box-sizing: border-box; }}
        textarea {{ height: 80px; resize: vertical; }}
        button {{ padding: 12px 30px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .status.think {{ background: #fff3cd; color: #856404; }}
        .status.generating {{ background: #d1ecf1; color: #0c5460; }}
        audio {{ width: 100%; margin-top: 5px; }}
        .settings {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
        h1 {{ margin-top: 0; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 LLM + CosyVoice 语音对话</h1>
        <div class="settings">
            <div class="row">
                <div>
                    <label>LLM API 地址 (OpenAI格式)：</label>
                    <input type="text" id="llm_url" value="http://127.0.0.1:8001/v1/chat/completions">
                </div>
                <div>
                    <label>参考音色：</label>
                    <select id="reference">
                        <option value="./asset/female.mp3">female.mp3</option>
                        <option value="./asset/dingzhen.mp3">dingzhen.mp3</option>
                        <option value="./asset/yujie.mp3">yujie.mp3</option>
                    </select>
                </div>
            </div>
            <div class="row" style="margin-top: 10px;">
                <div>
                    <label>System Prompt：</label>
                    <textarea id="system_prompt">你是一个友好的AI助手，请用简洁的中文回答用户的问题。</textarea>
                </div>
                <div>
                    <label>Prompt Text (TTS)：</label>
                    <input type="text" id="prompt_text" value="You are a helpful assistant. &lt;|endofprompt|&gt;">
                </div>
            </div>
            <div class="row" style="margin-top: 10px;">
                <div>
                    <label>Max Tokens：</label>
                    <input type="number" id="max_tokens" value="512" min="1" max="2048">
                </div>
                <div>
                    <label>Temperature：</label>
                    <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
                </div>
            </div>
            <div class="row" style="margin-top: 10px;">
                <div style="display: flex; align-items: center;">
                    <label style="margin-bottom: 0; margin-right: 10px;">自动播放语音：</label>
                    <input type="checkbox" id="autoplay" checked style="width: auto; height: 18px; width: 18px;">
                </div>
                <div></div>
            </div>
        </div>

        <div class="chat-container" id="chatContainer"></div>

        <div class="controls">
            <div>
                <label>你的消息：</label>
                <textarea id="message" placeholder="输入你的问题，按回车发送，Ctrl+回车换行..."></textarea>
            </div>
            <button id="sendBtn" onclick="sendMessage()">发送</button>
        </div>
        <div class="status" id="status"></div>
    </div>

<script>
const chatContainer = document.getElementById('chatContainer');
const statusDiv = document.getElementById('status');
let conversationHistory = [];

function addMessage(content, isUser) {{
    const div = document.createElement('div');
    div.className = 'message ' + (isUser ? 'user' : 'assistant');
    if (isUser) {{
        div.textContent = content;
    }} else {{
        div.innerHTML = '<div class="text">' + content + '</div>';
    }}
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}}

async function sendMessage() {{
    console.log('sendMessage clicked');
    const message = document.getElementById('message').value.trim();
    const llmUrl = document.getElementById('llm_url').value.trim();
    const reference = document.getElementById('reference').value;
    const systemPrompt = document.getElementById('system_prompt').value;
    const promptText = document.getElementById('prompt_text').value;
    const maxTokens = parseInt(document.getElementById('max_tokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const sendBtn = document.getElementById('sendBtn');

    if (!message) {{
        alert('请输入消息');
        return;
    }}

    sendBtn.disabled = true;

    addMessage(message, true);
    document.getElementById('message').value = '';
    conversationHistory.push({{role: 'user', content: message}});

    statusDiv.className = 'status think';
    statusDiv.textContent = '🧠 AI 正在思考...';

    try {{
        let fullPrompt = systemPrompt + '\\n\\n';
        conversationHistory.forEach(function(msg) {{
            if (msg.role === 'user') {{
                fullPrompt += 'User: ' + msg.content + '\\n';
            }} else {{
                fullPrompt += 'Assistant: ' + msg.content + '\\n';
            }}
        }});
        fullPrompt += 'Assistant:';

        console.log('Sending request', fullPrompt);
        const aiDiv = addMessage('', false);
        let fullContent = '';
        const textDiv = aiDiv.querySelector('.text');

        const response = await fetch('/chat/completion-stream', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
                message: fullPrompt,
                llm_url: llmUrl,
                system_prompt: systemPrompt,
                max_tokens: maxTokens,
                temperature: temperature,
                reference_audio_path: reference,
                prompt_text: promptText
            }})
        }});

        if (!response.ok) {{
            const data = await response.json();
            throw new Error(data.detail || '请求失败');
        }}

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {{
            const {{done, value}} = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\\n\\n');
            for (const line of lines) {{
                if (line.startsWith('data: ')) {{
                    const content = line.slice(6);
                    fullContent += content;
                    textDiv.textContent = fullContent;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }}
            }}
        }}

        conversationHistory.push({{role: 'assistant', content: fullContent}});

        const playBtn = document.createElement('button');
        playBtn.className = 'play-btn';
        playBtn.textContent = '🔊 流式播放';
        playBtn.onclick = async function() {{
            await playTextStream(fullContent, reference, promptText, this);
        }};
        textDiv.appendChild(document.createElement('br'));
        textDiv.appendChild(playBtn);

        // 自动播放
        const autoplayChecked = document.getElementById('autoplay').checked;
        if (autoplayChecked) {{
            statusDiv.textContent = '✅ 回复完成，自动播放中...';
            await playTextStream(fullContent, reference, promptText, playBtn);
        }} else {{
            statusDiv.textContent = '✅ 回复完成，请点击播放按钮听语音';
        }}

        chatContainer.scrollTop = chatContainer.scrollHeight;
        statusDiv.className = 'status';
    }} catch (error) {{
        console.error('Error:', error);
        statusDiv.textContent = '❌ 错误：' + error.message;
        statusDiv.className = 'status think';
    }} finally {{
        sendBtn.disabled = false;
    }}
}}

async function playTextStream(fullText, reference, promptText, button) {{
    console.log('playTextStream started with howler.js', fullText);
    button.disabled = true;
    button.textContent = '🔊 流式生成中...';
    const container = button.parentNode;

    container.appendChild(document.createElement('br'));

    // 添加提示信息
    const infoDiv = document.createElement('div');
    infoDiv.style.color = '#666';
    infoDiv.style.fontSize = '12px';
    infoDiv.style.marginTop = '5px';
    infoDiv.textContent = '🔊 howler.js流式播放中，边生成边播放...';
    container.appendChild(infoDiv);

    try {{
        const sampleRate = {sample_rate};
        console.log('sample rate:', sampleRate);

        // 使用howler.js + AudioContext流式播放
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({{sampleRate: sampleRate}});
        await audioContext.resume();

        // howl实例
        let sound = null;
        let audioBufferQueue = [];
        let currentOffset = 0;
        let source = null;

        // 预分配缓冲区 - howler.js会更好管理内存
        const BUFFER_DURATION = 2; // seconds
        const totalSamplesEstimate = Math.floor(sampleRate * (fullText.length / 4)); // ~4 chars per second
        const audioBuffer = audioContext.createBuffer(1, totalSamplesEstimate, sampleRate);

        const params = new URLSearchParams({{
            text: fullText,
            reference_audio_path: reference,
            prompt_text: promptText
        }});

        const response = await fetch('/chat/tts-pcm?' + params.toString());
        console.log('got response', response.ok);

        if (!response.ok) {{
            const err = await response.json();
            throw new Error(err.detail || '生成失败');
        }}

        const reader = response.body.getReader();
        let startedPlayback = false;
        let totalSamplesReceived = 0;
        let byteBuffer = new Uint8Array(0);

        // 使用howler.js播放
        function startPlayback() {{
            if (startedPlayback) return;
            startedPlayback = true;

            // 创建source
            source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();
            console.log('Playback started with howler.js approach');

            statusDiv.textContent = '🔊 正在流式播放...';
            statusDiv.className = 'status generating';
        }}

        // 保存所有收到的数据用于生成完整WAV
        let allInt16Data = [];
        let totalInt16Length = 0;

        async function processNextChunk() {{
            const {{done, value}} = await reader.read();
            if (done) {{
                console.log('stream complete, total samples', totalSamplesReceived);
                button.textContent = '✅ 流式生成完成';
                statusDiv.textContent = '✅ 流式生成完成';
                statusDiv.className = 'status';

                // 创建完整的WAV文件并添加可重播播放器
                createWavPlayer();
                button.disabled = false;
                return;
            }}

            // 累积字节，确保对齐到2字节边界 (int16)
            const newBuffer = new Uint8Array(byteBuffer.length + value.byteLength);
            newBuffer.set(byteBuffer, 0);
            newBuffer.set(value, byteBuffer.length);
            byteBuffer = newBuffer;

            // 处理完整的int16样本
            const completeBytes = byteBuffer.length - (byteBuffer.length % 2);
            if (completeBytes > 0) {{
                const int16Buffer = new Int16Array(byteBuffer.buffer.slice(0, completeBytes));
                // 保存数据用于后续创建完整WAV
                allInt16Data.push(new Int16Array(int16Buffer));
                totalInt16Length += int16Buffer.length;

                // 转换为float32 for WebAudio
                const floatSamples = new Float32Array(int16Buffer.length);
                for (let i = 0; i < int16Buffer.length; i++) {{
                    floatSamples[i] = int16Buffer[i] / 32768.0;
                }}

                // 复制到音频缓冲区
                audioBuffer.copyToChannel(floatSamples, 0, currentOffset);
                currentOffset += floatSamples.length;
                totalSamplesReceived += floatSamples.length;

                // 剩余字节留在缓冲区
                const remainingBytes = byteBuffer.length - completeBytes;
                if (remainingBytes > 0) {{
                    byteBuffer = byteBuffer.slice(completeBytes);
                }} else {{
                    byteBuffer = new Uint8Array(0);
                }}

                if (!startedPlayback && currentOffset >= sampleRate * 0.2) {{
                    // 积累了0.2秒数据后开始播放
                    startPlayback();
                }}
            }}

            processNextChunk();
        }};

        // 创建完整WAV并添加播放器
        function createWavPlayer() {{
            // 合并所有int16数据
            const combinedInt16 = new Int16Array(totalInt16Length);
            let offset = 0;
            for (const chunk of allInt16Data) {{
                combinedInt16.set(chunk, offset);
                offset += chunk.length;
            }}

            // 转换为float32
            const combinedFloat = new Float32Array(combinedInt16.length);
            for (let i = 0; i < combinedInt16.length; i++) {{
                combinedFloat[i] = combinedInt16[i] / 32768.0;
            }}

            // 编码为WAV
            const wavBlob = encodeWav(combinedFloat, sampleRate);
            const audioUrl = URL.createObjectURL(wavBlob);

            // 添加播放器
            const audioPlayer = document.createElement('div');
            audioPlayer.style.marginTop = '10px';
            audioPlayer.innerHTML = '<div style="font-size: 12px; color: #666; margin-bottom: 5px;">🔊 完整音频（可重复播放）：</div>';
            const audio = document.createElement('audio');
            audio.src = audioUrl;
            audio.controls = true;
            audio.style.width = '100%';
            audioPlayer.appendChild(audio);
            container.appendChild(audioPlayer);
        }}

        // 编码WAV文件
        function encodeWav(samples, sampleRate) {{
            const bytesPerSample = 2; // int16
            const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
            const view = new DataView(buffer);

            // RIFF identifier
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * bytesPerSample, true);
            writeString(view, 8, 'WAVE');
            // fmt subchunk
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 1 * bytesPerSample, true);
            view.setUint16(32, 1 * bytesPerSample, true);
            view.setUint16(34, 8 * bytesPerSample, true);
            // data subchunk
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * bytesPerSample, true);

            // write samples
            let offset = 44;
            for (let i = 0; i < samples.length; i++, offset += 2) {{
                const sample = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            }}

            function writeString(view, offset, string) {{
                for (let i = 0; i < string.length; i++) {{
                    view.setUint8(offset + i, string.charCodeAt(i));
                }}
            }}

            return new Blob([view], {{type: 'audio/wav'}});
        }}

        processNextChunk();

    }} catch (error) {{
        console.error('TTS error:', error);
        statusDiv.textContent = '❌ 生成语音失败：' + error.message;
        statusDiv.className = 'status think';
        button.disabled = false;
        button.textContent = '🔊 流式播放';
    }}
}}

document.getElementById('message').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter' && !(e.ctrlKey || e.metaKey)) {{
        e.preventDefault();
        sendMessage();
    }}
}});

addMessage('你好！我是AI助手，请输入你的问题，我会用语音回答你！', false);
conversationHistory.push({{role: 'assistant', content: '你好！我是AI助手，请输入你的问题，我会用语音回答你！'}});
</script>
</body>
</html>'''
    return HTMLResponse(content=html)

@app.post("/chat/completion-stream", summary="调用LLM生成文字回复（流式输出）")
async def chat_completion_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        lines = request.message.split('\n')
        messages = []

        system_prompt = lines[0].strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('User:'):
                messages.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith('Assistant:'):
                content = line[10:].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})

        if not messages or messages[-1]['role'] != 'user':
            if messages:
                last_user = request.message.split('User:')[-1].split('\n')[0].strip()
                messages.append({"role": "user", "content": last_user})

        base_url = request.llm_url
        if '/v1/chat/completions' in base_url:
            base_url = base_url.rsplit('/v1/chat/completions', 1)[0]
        elif '/chat/completions' in base_url:
            base_url = base_url.rsplit('/chat/completions', 1)[0]

        client = OpenAI(
            base_url=base_url,
            api_key="dummy"
        )

        def sync_generate():
            full_content = ""
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    yield f"data: {content_chunk}\n\n"

        async def generate():
            for chunk in sync_generate():
                yield chunk

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败，请检查LLM服务是否启动: {str(e)}")

@app.get("/chat/tts-pcm", summary="CosyVoice原生流式输出PCM音频数据，howler.js边接收边播放")
async def chat_tts_pcm(text: str, reference_audio_path: str = "./asset/female.mp3", prompt_text: str = "You are a helpful assistant. <|endofprompt|>"):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model not loaded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not os.path.exists(reference_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {reference_audio_path}")

    try:
        # 直接流式输出float32 PCM数据，howler.js在前端处理播放
        def generate():
            for chunk in cosyvoice.inference_instruct2(
                text,
                prompt_text,
                reference_audio_path,
                stream=True
            ):
                speech_np = chunk['tts_speech'].squeeze(0).cpu().numpy()
                # Convert float32 to int16 for better compatibility
                speech_int16 = (speech_np * 32767).astype(np.int16)
                yield speech_int16.tobytes()

        return StreamingResponse(
            generate(),
            media_type="audio/pcm"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {str(e)}")

@app.get("/chat/tts-full", summary="将文本生成完整语音")
async def chat_tts_full(text: str, reference_audio_path: str = "./asset/female.mp3", prompt_text: str = "You are a helpful assistant. <|endofprompt|>"):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model not loaded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not os.path.exists(reference_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio file not found: {reference_audio_path}")

    try:
        audio_buffer = io.BytesIO()
        full_audio = None
        for chunk in cosyvoice.inference_instruct2(
            text,
            prompt_text,
            reference_audio_path,
            stream=True
        ):
            speech_np = chunk['tts_speech'].squeeze(0).cpu().numpy()
            if full_audio is None:
                full_audio = speech_np
            else:
                full_audio = np.concatenate([full_audio, speech_np])

        sf.write(audio_buffer, full_audio, cosyvoice.sample_rate, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS生成失败: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if cosyvoice is not None else "unhealthy",
        "model_loaded": cosyvoice is not None,
        "sample_rate": cosyvoice.sample_rate if cosyvoice else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
