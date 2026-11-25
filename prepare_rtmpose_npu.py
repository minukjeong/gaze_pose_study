"""Convert the cached RTMPose-S ONNX to a static-batch FP16 OpenVINO IR."""
from pathlib import Path
from openvino import Core, save_model


def main():
    source = (Path.home()/'.cache'/'rtmlib'/'hub'/'checkpoints'/
              'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx')
    output = Path(__file__).resolve().parent/'models'/'rtmpose_npu'/'rtmpose_s_static.xml'
    if not source.is_file():
        raise FileNotFoundError(f'Run RTMLib once to download {source.name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    core = Core()
    model = core.read_model(source)
    model.reshape({model.input(0): [1, 3, 256, 192]})
    save_model(model, output, compress_to_fp16=True)
    print('SAVED', output)


if __name__ == '__main__':
    main()
