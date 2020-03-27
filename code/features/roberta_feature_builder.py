import csv
import copy
import json
import os
import numpy as np
import pandas as pd

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r",errors='ignore') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DPTProcessor(DataProcessor):
    """Processor for the DOT data set"""

    def get_examples(self,data_dir,set_type,code=''):
        lines = self.read_csv(data_dir+'/'+set_type+'.csv')
        examples = self.create_examples(lines,set_type,code)
        return(examples)

    def get_labels(self,code=''):
        """See base class."""
        data_codes = range(7)
        people_codes = range(9)
        things_codes = range(8)
        labels = [str(d)+str(p)+str(t) for d in data_codes for p in people_codes for t in things_codes]
        labels = [None] if code else labels
        return labels

    def read_csv(self,input_file):
        df = pd.read_csv(input_file)
        lines = df.to_numpy().tolist()
        return(lines)

    def create_examples(self,lines,set_type,code=''):
        examples = []
        code_to_idx = {'data':0,'people':1,'things':2,'':None}
        label_idx = code_to_idx[code]
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            label = line[3][1:]
            label = label[label_idx] if code else label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def convert_examples_to_features(self,examples, tokenizer,output_mode,
                                          max_length=190,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True):
        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
        Returns:
            If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """

        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):

            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            if output_mode == 'classification':
                label = label_map[example.label]
            elif output_mode == 'regression':
                label = float(example.label)

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label))

        return features

class AttributesProcessor(DataProcessor):
    """Processor for the DOT data set"""

    def get_examples(self,data_dir,set_type,code=''):
        lines = self.read_csv(data_dir+'/'+set_type+'.csv')
        examples = self.create_examples(lines,set_type,code)
        return(examples)

    def get_labels(self,code=''):
        GED = np.arange(1.5,6.5,0.5)
        coord = [1.0,2.5,3.0,3.5,4.0,4.5,5.0]
        finger_dext = np.arange(1.5,5,0.5)
        sts = [0,1]
        dcp = [0,1]
        labels = [(str(G),str(c),str(f),str(s),str(d))
        for G in GED for c in coord for f in finger_dext for s in sts for d in dcp]
        labels = [None] if code else labels
        return labels

    def read_csv(self,input_file):
        df = pd.read_csv(input_file)
        lines = df.to_numpy().tolist()
        return(lines)

    def create_examples(self,lines,set_type,code=''):
        examples = []
        code_to_idx = {'GED':0,'EHFCoord':1,'FingerDexterity':2,'DCP':3,'STS':4}
        label_idx = code_to_idx[code]
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            label = (line[5],line[6],line[7],line[8],line[9])
            label = tuple(str(i) for i in label)
            label = label[label_idx] if code else label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def convert_examples_to_features(self,examples, tokenizer, output_mode,
                                          max_length=190,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True):
        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
        Returns:
            If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """

        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            if output_mode == 'classification':
                label = label_map[example.label]
            elif output_mode == 'regression':
                label = float(example.label)

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label))

        return features
