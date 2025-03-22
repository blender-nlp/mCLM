



from transformers.processing_utils import ProcessingKwargs, ProcessorMixin

from transformers.modeling_utils import PreTrainedModel


from mCLM.data.processing import smiles_to_mol

from transformers import PreTrainedTokenizer, AddedToken



class mCLMProcessor(ProcessorMixin):
    r"""
    """

    def __init__(
        self,
        molecule_tokenizer=None,
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

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.const_helper = self.build_const_helper()

    #@torch.no_grad()
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
        return BatchFeature(data={**text_inputs, "mol_mask": size_list}, tensor_type=kwargs.get("return_tensors"))

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

        molecule_tokens = []
        for mol in molecule:
            molecule_input = [smiles_to_mol(m) for m in mol.split('.')]
            molecule_input = [m.to(self.molecule_tokenizer.device, self.molecule_tokenizer.dtype) for m in molecule_input]
            molecule_tokens.append(self.molecule_tokenizer.encode(molecule_input).squeeze(0))

        return molecule_tokens





class mCLMGNNPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    #config_class = Emu3VisionVQConfig
    #base_model_prefix = "emuvideovq"
    #main_input_name = "pixel_values"
    #_no_split_modules = ["Emu3VisionVQResnetBlock", "Emu3VisionVQAttnBlock", "Emu3VisionVQResnetTemporalBlock"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class mCLMGNNModel(mCLMGNNPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.mol_encoder = GNNMolEncoder(
            node_dim=self.config["node_dim"],
            edge_dim=self.config["edge_dim"],
            hidden_dim_graph=self.config["hidden_dim_graph"],
            hidden_dim_ffn=None,
            num_mp_layers=self.config["num_mp_layers"],
            out_channels=config["latent_size"],
            dropout=self.config["dropout"],
            num_readout_layers=1,
            mol_features_size=0,
            aggr=self.config["aggr"],
            jk=self.config["jk"],
        )

        self.post_init()

    def encode(self, x: torch.Tensor):

        h = self.encoder(x)

        return h

    def decode(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 3:
            x = x.unsqueeze(1)

        b, t, h, w = x.shape
        quant = self.quantize.embedding(x.flatten())
        c = quant.shape[-1]
        quant = quant.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        quant2 = self.post_quant_conv(quant)

        quant = quant.permute(0, 2, 1, 3, 4)
        quant2 = quant2.permute(0, 2, 1, 3, 4)

        video = self.decoder(quant2, quant)
        video = video.reshape(
            b,
            t * self.config.temporal_downsample_factor,
            self.config.out_channels,
            h * self.spatial_scale_factor,
            w * self.spatial_scale_factor,
        )
        if ndim == 3:
            return video[:, 0]
        return video

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype



