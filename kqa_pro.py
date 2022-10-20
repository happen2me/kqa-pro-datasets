"""KQA Pro: A large-scale, diverse, challenging dataset of complex question answering over knowledge base."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{KQAPro,
    title={{KQA P}ro: A Large Diagnostic Dataset for Complex Question Answering over Knowledge Base},
    author={Cao, Shulin and Shi, Jiaxin and Pan, Liangming and Nie, Lunyiu and Xiang, Yutong and Hou, Lei and Li, Juanzi and He, Bin and Zhang, Hanwang},
    booktitle={ACL'22},
    year={2022}
    }
"""

_DESCRIPTION = """\
A large-scale, diverse, challenging dataset of complex question answering over knowledge base.
"""

_URL = "https://thukeg.gitee.io/kqa-pro/"
_DOWNLOAD_URL = "https://cloud.tsinghua.edu.cn/f/df54ff66d1dc4ca7823e/?dl=1"

_TRAIN_CONFIG_NAME = "train_val"
_TEST_CONFIG_NAME = "test"

class KQAProConfig(datasets.BuilderConfig):
    """BuilderConfig for KQA Pro."""

    def __init__(self, **kwargs):
        """BuilderConfig for KQA Pro.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(KQAProConfig, self).__init__(**kwargs)


class KQAPro(datasets.GeneratorBasedBuilder):
    """KQAPro: A large scale knowledge-based question answering dataset."""

    BUILDER_CONFIGS = [
        KQAProConfig(
            name=_TRAIN_CONFIG_NAME,
            description="KQA Pro",
            data_dir="data"
        ),
        KQAProConfig(
            name=_TEST_CONFIG_NAME,
            description="KQA Pro",
            data_dir="data"
        ),
    ]


    def _info(self):
        if self.config.name == _TEST_CONFIG_NAME:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "question": datasets.Value("string"),
                        "choices": datasets.features.Sequence(datasets.Value("string")),
                    }
                ),
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "sparql": datasets.Value("string"),
                    "program": datasets.features.Sequence(
                        {
                            "function": datasets.Value("string"),
                            "dependencies":  datasets.features.Sequence(datasets.Value("int32")),
                            "inputs": datasets.features.Sequence(datasets.Value("string"))
                        }
                    ),
                    "choices": datasets.features.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string")
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        data_dir = os.path.join(download_dir, self.config.data_dir)
        downloaded_files = {
            "train": os.path.join(data_dir, "train.json"),
            "val": os.path.join(data_dir, "val.json"),
            "test": os.path.join(data_dir, "test.json")
        }

        if self.config.name == _TEST_CONFIG_NAME:
            return [
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                        "filepath": downloaded_files["test"]})
            ]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": downloaded_files["val"]})
        ]


    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            kqa = json.load(f)
            for idx, sample in enumerate(kqa):
                yield idx, sample
