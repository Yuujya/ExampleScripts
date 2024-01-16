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

    def process_transcription_request(self, prompt: str):
        response = self.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=self.temperature
        )
        content = response.choices[0].message.content
        return content

    def from_transcription_to_speech(self, filename, transcription):
        """
        Generate spoken audio from transcription.
        :return: string prompt of generated voice
        """
        speech_file_path = filename
        response = self.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=transcription
        )
        response.stream_to_file(speech_file_path)


def record_voice(filename: str = 'output.wav',
                 seconds=3):
    fs = 44100  # Sample rate
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, recording)
    return filename
