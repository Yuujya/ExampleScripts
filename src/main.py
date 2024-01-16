from client import Client
from playsound import playsound

def main():
    client = Client()
    while True:
        print(f'Please choose a query type: '
              f'(1 - chat request,'
              f' 2 - image generation,'
              f' 3 - voice transcription,'
              f' q - quit program)')
        choice = input()
        if choice == '1' or choice == '2':
            print(f'Please input a prompt.')
            prompt = input()
            match choice:
                case '1':
                    client.stream_chat_request(prompt)
                case '2':
                    url = client.generate_image(prompt)
                    print(f'image url = {url}')
        elif choice == '3':
            transcription = client.create_transcription()
            print(transcription)
            response_from_transcription = client.process_transcription_request(transcription)
            print(response_from_transcription)
            from_transcription_file = "from_transcription.wav"
            client.from_transcription_to_speech(from_transcription_file, response_from_transcription)
            playsound(from_transcription_file)

        elif choice == 'q':
            print(f'Exit program.')
            break
        else:
            print(f'Invalid input!')
        print()


if __name__ == '__main__':
    main()
