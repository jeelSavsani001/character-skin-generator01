# Character Skin Generator
Various models are tried to generate new and innovative skins for the game characters like valorant , brawlstars , fortnite , etc.

Following are the models which we have tested:
1) timbrooks/instruct-pix2pix 
2) InstantX/FLUX.1-dev-Controlnet-Canny(It does not work on less VRAM in full original form so we have put Demo results
   from https://huggingface.co/spaces/fffiloni/FLUX.1_dev_InstantX_ControlNet_Canny)
3) LoRA finetuned diffusers/sdxl-instructpix2pix-768(only for 1 epoch)

Results were good on basic cartoonish characters like the following:

![Original_Image.](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Bea_Original.jpg)


![Skin_1](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Res3.jpg)


![Skin_2](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Res5.jpg)


These were mostly timbrooks/instruct-pix2pix 

Then we wanted to try this out on complex characters where it failed badly but we finetuned a LoRA on SDXL using the dataset https://huggingface.co/datasets/olly4/fortnite_characters using the script from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md and we got following results:

![Original_Image.](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Fortnite.jpg)

![Skin1](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Fortnite_skin.jpg)

![Skin2](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Fortnite_2.jpg)



And finally we wanted to try it out on valorant characters, and results were similar to above but flux gave us some better results as follow:

![Original_Image.](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Valo.jpg)

![Skin1](https://github.com/jeelSavsani001/character-skin-generator01/blob/main/Res4.jpg)

