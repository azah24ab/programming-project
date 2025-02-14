import os
import json
from rvc import Config, load_hubert, get_vc, rvc_infer

def convert_voice(wav_path, pth_path, index_path, temp_dir='temp', models_dir='models_folder'):
    try:
        device = "cuda:0"
        config = Config(device, True)
        hubert_model = load_hubert(device, config.is_half, os.path.join(models_dir, 'hubert_base.pt'))
        cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, pth_path)

        # Voice conversion using the model index, pth and wav file selected
        output_audio_path = os.path.join(temp_dir, 'output.wav')
        f0up_key = 0
        f0_method = "rmvpe"
        index_rate = 0.5
        filter_radius = 3
        rms_mix_rate = 0.25
        protect = 0.33
        crepe_hop_length = 128
        
        wav_opt = rvc_infer(
            index_path, 
            index_rate=index_rate,
            input_path=wav_path,
            output_path=output_audio_path,
            pitch_change=f0up_key,
            f0_method=f0_method,
            cpt=cpt,
            version=version,
            net_g=net_g,
            filter_radius=filter_radius,
            tgt_sr=tgt_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            crepe_hop_length=crepe_hop_length,
            vc=vc,
            hubert_model=hubert_model
        )
        
        return output_audio_path
        
    except Exception as e:
        print(f"Error converting voice: {str(e)}")
        return None

def main():
    try:
        with open('conversion_config.json', 'r') as f:
            config = json.load(f)
        
        # Paths for the wav, index, and pth we stored in the config file 
        wav_path = config['wav_path']
        index_path = config['index_path']
        pth_path = config['pth_path']
        
        print("Starting voice conversion with:")
        print(f"WAV file: {wav_path}")
        print(f"Index file: {index_path}")
        print(f"PTH file: {pth_path}")
        
        # Calling the convert_voice function in order to initiate the vocal conversion using the model selected
        output_path = convert_voice(wav_path, pth_path, index_path)
        
        if output_path:
            print(f"Voice conversion successful! Output saved to: {output_path}")
        else:
            print("Voice conversion failed!")
            
    except Exception as e:
        print(f"Error during voice conversion: {str(e)}")
        return None

if __name__ == "__main__":
    main()
