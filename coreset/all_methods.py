import torch
import numpy as np

class Base:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def select(self):
        return

class KCenter(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(KCenter, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            # dis = distance(feature, mean)[:,0]
            dis = torch.cdist(feature, mean)[:,0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt-1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)

            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)


class Herding(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Herding, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        # herding # class by class
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)


class Random(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Random, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train

        labels_train = self.data.labels_train
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            selected = np.random.permutation(idx)
            idx_selected.append(selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)


