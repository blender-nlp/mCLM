



from transformers.processing_utils import ProcessingKwargs, ProcessorMixin



class mCLMProcessor(ProcessorMixin):
    r"""
    """

    def __init__(
        self,
        image_processor=None,
        vision_tokenizer=None,
        text_tokenizer=None,
        chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
        prefix_template="{H}*{W}",
        visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>"),
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"

        self.vision_tokenizer = vision_tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        self.vis_tok_spatial_factor = 2 ** (len(self.vision_tokenizer.config.ch_mult) - 1)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.const_helper = self.build_const_helper()

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        molecule: Optional[Image.Image | List[Image.Image]] = None,
        *,
        mode: str = "G",
        padding_mol: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        """
        assert mode in ('G', 'U'), "mode must be 'G' or 'U'."
        if isinstance(text, str):
            text = [text]

        if isinstance(image, Image.Image):
            image = [image]

        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        image_tokens = None
        if mode == 'G':
            if image is not None:
                raise ValueError("You have to specify only `text` in generation mode")

            if isinstance(ratio, str):
                ratio = [ratio] * len(text)

            if len(ratio) != len(text):
                raise ValueError("ratio number must match text number")
        else:
            if image is None:
                raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

            if not isinstance(image, Sequence) and not isinstance(image, Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            if isinstance(image, Sequence) and not isinstance(image[0], Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            image_tokens = self.tokenize_image(image, padding_image=padding_image)
            if len(text) != len(image_tokens):
                raise ValueError("number of image must match number of text prompt")

        prompt_list, size_list = [], []
        for idx, text_prompt in enumerate(text):
            prompt = self.tokenizer.bos_token
            if mode == 'U':
                h, w = image_tokens[idx].shape
                imgstr = self.to_imgstr(image_tokens[idx])
                image_prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token + 
                    imgstr +
                    self.tokenizer.eol_token +
                    self.tokenizer.eof_token +
                    self.tokenizer.eoi_token
                )
                prompt += self.chat_template.format(image_prompt=image_prompt, text_prompt=text_prompt)
            else:
                h, w = self.calculate_generate_size(ratio[idx], image_area, self.vision_tokenizer.spatial_scale_factor)
                image_prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token
                )
                prompt += (text_prompt + image_prompt)

            prompt_list.append(prompt)
            size_list.append([h, w])

        text_inputs = self.tokenizer(prompt_list, **kwargs)
        return BatchFeature(data={**text_inputs, "image_size": size_list}, tensor_type=kwargs.get("return_tensors"))

    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        pass
        docs = self.tokenizer.batch_decode(*args, **kwargs)
        return [self.multimodal_decode(d) for d in docs]

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode(doc)

    @torch.no_grad()
    def vision_encode(self, *args, **kwargs):
        return self.vision_tokenizer.encode(*args, **kwargs)

    @torch.no_grad()
    def vision_decode(self, *args, **kwargs):
        return self.vision_tokenizer.decode(*args, **kwargs)

    @torch.no_grad()
    def multimodal_decode(self, doc):
        multimodal_output = []
        pattern = rf'({re.escape(self.tokenizer.boi_token)}.*?{re.escape(self.tokenizer.eoi_token)})'
        chunks = re.split(pattern, doc)
        for c in chunks:
            if len(c) == 0:
                continue

            if self.tokenizer.boi_token in c:
                image = []
                image_rows = re.split(re.escape(self.tokenizer.eol_token), c)
                for r in image_rows:
                    token_ids = re.findall(self.visual_template[1], r)
                    if len(token_ids) > 0:
                        row_token = [int(m) for m in token_ids]
                        image.append(row_token)
                image = torch.tensor(image, dtype=torch.long, device=self.vision_tokenizer.device)
                image = self.vision_tokenizer.decode(image[None]).float()
                image = self.image_processor.postprocess(image)["pixel_values"][0]
                multimodal_output.append(image)
            else:
                multimodal_output.append(c)

        return multimodal_output if len(multimodal_output) > 1 else multimodal_output[0]


    def tokenize_molecule(self, molecule: List[str], padding_molecule: bool = False):
        is_all_same_size, prev_size = True, None
        for im in image:
            if prev_size is not None:
                is_all_same_size &= (prev_size == im.size)
            prev_size = im.size

        if is_all_same_size:
            image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
            image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
        elif padding_image:
            image_inputs = [self.image_processor(im, return_tensors="pt")["pixel_values"] for im in image]
            image_shapes = [im.shape[2:] for im in image_inputs]
            max_shape = (
                max([im_shape[0] for im_shape in image_shapes]),
                max([im_shape[1] for im_shape in image_shapes]),
            )
            image_inputs = [
                F.pad(im_inp, (0, max_shape[1] - im_shape[1], 0, max_shape[0] - im_shape[0]))
                for im_inp, im_shape in zip(image_inputs, image_shapes)
            ]
            image_inputs = torch.cat(image_inputs, dim=0).to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
            image_tokens = [
                im_tok[:ceil(im_shape[0] / self.vis_tok_spatial_factor), :ceil(im_shape[1] / self.vis_tok_spatial_factor)]
                for im_tok, im_shape in zip(image_tokens, image_shapes)
            ]
        else:
            image_tokens = []
            for im in image:
                image_input = self.image_processor(im, return_tensors="pt")["pixel_values"]
                image_input = image_input.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                image_tokens.append(self.vision_tokenizer.encode(image_input).squeeze(0))

        return image_tokens








