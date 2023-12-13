from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write


class Client(OpenAI):
    def __init__(self, model='gpt-3.5-turbo', temperature=0.0):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def stream_chat_request(self, prompt: str, stream=True):
        stream = self.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=stream,
            temperature=self.temperature
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end='')

    def generate_image(self, prompt):
        response = self.images.generate(
            model='dall-e-3',
            prompt=prompt,
            n=1,
            size='1024x1024'
        )
        return response.data[0].url

    def create_transcription(self):
        """
        Record voice and return a string prompt.
        :return: string prompt of recorded voice
        """
        filename = record_voice()
        audio_file = open(filename, 'rb')
        transcript = self.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file
        )
        return transcript.text


def record_voice(filename: str = 'output.wav',
                 seconds=3):
    fs = 44100  # Sample rate
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, recording)
    return filename
