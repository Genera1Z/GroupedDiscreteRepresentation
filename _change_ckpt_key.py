import torch as pt


def main():
    ckpt_file = "/home/GeneralZ/Downloads/slotdiffusion-gdr-coco/slotdiffusion_r_vqvae-coco-gdr/best.pth"
    ckpt_file2 = "/home/GeneralZ/Downloads/slotdiffusion-gdr-coco/slotdiffusion_r_vqvae-coco-gdr/best2.pth"
    state_dict = pt.load(ckpt_file)
    state_dict2 = {}
    for k, v in state_dict.items():
        if k == "m.decode.posit_embed.pe":
            k = "m.decode.posit_embed._pe"
        state_dict2[k] = v
    pt.save(state_dict2, ckpt_file2)


if __name__ == "__main__":
    main()
