# skeleton source from https://research.wmz.ninja/attachments/articles/2018/03/jigsaw_cifar100.html

import torch
import torchvision
import torchvision.transforms as transforms
import math
import argparse
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(86201)

# CIFAR images are 32x32
L = 3
L2 = L * L
BL = 32 // L
perm_inds = []
for i in range(L):
    for j in range(L):
        perm_inds.append((i * BL, j * BL))

# Simply maps each pixel to [-1, 1]
img_mean = 0.5
img_std = 0.5


def permuteLxL(images):
    """
    Splits the images into LxL pieces and randomly permutes the pieces.
    """
    p_images = torch.FloatTensor(images.size())
    perms = torch.LongTensor(images.size()[0], L2)
    for i in range(images.size()[0]):
        p = torch.randperm(L2)
        for j in range(L2):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[p[j]]
            p_images[i, :, tr:tr + BL, tc:tc + BL] = images[i, :, sr:sr + BL, sc:sc + BL]
        perms[i, :] = p
    return (p_images, perms)


def subsqaureLxL(image, ri, rj):
    """
    return (ri, rj) sub-square
    """
    ri *= BL
    rj *= BL
    return image[:, ri:ri+BL, rj:rj+BL]


def restoreLxL(p_images, perms):
    """
    Restores the original image from the pieces and the given permutation.
    """
    images = torch.FloatTensor(p_images.size())
    for i in range(images.size()[0]):
        for j in range(L2):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[int(perms[i][j])]
            images[i, :, sr:sr + BL, sc:sc + BL] = p_images[i, :, tr:tr + BL, tc:tc + BL]
    return images


def perm2vecmatLxL(perms):
    """
    Converts permutation vectors to vectorized assignment matrices.
    """
    n = perms.size()[0]
    mat = torch.zeros(n, L2, L2)
    # m[i][j] : i is assigned to j
    for i in range(n):
        for k in range(L2):
            mat[i, k, perms[i, k]] = 1.
    return mat.view(n, -1)


def vecmat2permLxL(x):
    """
    Converts vectorized assignment matrices back to permutation vectors.
    Note: this function is compatible with GPU tensors.
    """
    n = x.size()[0]
    x = x.view(n, L2, L2)
    _, ind = x.max(2)
    return ind


def imshow(img, title=None):
    """
    Displays a torch image.
    """
    img = img * img_std + img_mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title != None:
        plt.title(title)


batch_size = 32
dataset_dir = './data'

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((img_mean, img_mean, img_mean), (img_std, img_std, img_std))])

train_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)

# Plot some training samples.
sample_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
dataiter = iter(sample_loader)
images, labels = next(dataiter)

p_images, perms = permuteLxL(images)

# Check the implementation of per2vecmat and vecmat2perm.
assert (vecmat2permLxL(perm2vecmatLxL(perms)).equal(perms))

'''
# Show permuted images.
plt.figure()
imshow(torchvision.utils.make_grid(p_images))
plt.show()
# Show restored images.
plt.figure()
imshow(torchvision.utils.make_grid(restoreLxL(p_images, perms)))
plt.show()
'''

# Prepare training, validation, and test samples.
validation_ratio = 0.1
total = len(train_set)
ind = list(range(total))
n_train = int(np.floor((1. - validation_ratio) * total))
train_ind, validation_ind = ind[:n_train], ind[n_train:]
train_subsampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
validation_subsampler = torch.utils.data.sampler.SubsetRandomSampler(validation_ind)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_subsampler, num_workers=0)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=validation_subsampler, num_workers=0)

print('Number of training batches: {}'.format(len(train_loader)))
print('Number of validation batches: {}'.format(len(validation_loader)))

test_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


class RectConvNet_3x3(nn.Module):
    def __init__(self):
        super().__init__()
        # for L = 3
        # 3 x 10 x 20
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        nn.init.xavier_normal_(self.conv1.weight)
        # 10 x 10 x 20
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        nn.init.xavier_normal_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(10)
        # 10 x 10 x 20
        self.pool1 = nn.MaxPool2d(2, 2)
        # 10 x 5 x 10
        self.conv3 = nn.Conv2d(10, 20, 3)
        nn.init.xavier_normal_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(20)
        # 20 x 3 x 8
        self.pool2 = nn.MaxPool2d(2, 2, padding=(1, 0))
        # 20 x 2 x 4
        self.fc1 = nn.Linear(20 * 2 * 4, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(128)
        # 80
        self.fc2 = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(128)
        # 8
        self.fc3 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc3.weight)
        # self.fc3_bn = nn.BatchNorm1d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(-1, 20 * 2 * 4)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        # x = F.softmax(self.fc3_bn(self.fc3(x)), dim=1)
        x = torch.sigmoid(self.fc3(x))
        return x



class RectConvNet_2x2(nn.Module):
    def __init__(self):
        super().__init__()
        # for L = 2
        # 3 x 16 x 32
        self.conv1 = nn.Conv2d(3, 8, 3)
        # 8 x 14 x 30
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv2_bn = nn.BatchNorm2d(8)
        # 8 x 12 x 28
        self.pool1 = nn.MaxPool2d(2, 2)
        # 8 x 6 x 14
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv3_bn = nn.BatchNorm2d(16)
        # 16 x 4 x 12
        self.pool2 = nn.MaxPool2d(2, 2)
        # 16 x 2 x 6
        self.fc1 = nn.Linear(192, 80)
        self.fc1_bn = nn.BatchNorm1d(80)
        # 80
        self.fc2 = nn.Linear(80, 8)
        self.fc2_bn = nn.BatchNorm1d(8)
        # 8
        self.fc3 = nn.Linear(8, 2)
        self.fc3_bn = nn.BatchNorm1d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(-1, 16 * 2 * 6)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.softmax(self.fc3_bn(self.fc3(x)), dim=1)
        return x


def check_adjust(p1, p2):
    p1v = [p1 // L, p1 % L]
    p2v = [p2 // L, p2 % L]
    return (p1v[0] == p2v[0]) & (p1v[1] + 1 == p2v[1])


# BL x 2BL
def pre_process(imgs):
    pass
    '''
    for j in range(2*BL):
        v = 1.0 / max(BL+1-j, j-BL)
        imgs[:, :, :, j] *= v
    '''


def get_rand(l, r):
    return torch.randint(l, r, (1,)).item()


def get_index(prob):
    target = torch.rand(1).item()
    p1 = 0
    p2 = 0
    if target < prob:
        p1 = get_rand(0, L) * L + get_rand(0, L - 1)
        p2 = p1 + 1
    else:
        p1 = torch.randint(0, L2, (1,)).item()
        p2 = p1
        while p2 == p1:
            p2 = torch.randint(0, L2, (1,)).item()
    return p1, p2


# Training process
def train_model(model, criterion, optimizer, train_loader, validation_loader,
                n_epochs=40, save_file_name=None):
    loss_history = []
    val_loss_history = []
    acc_history = []
    val_acc_history = []
    for epoch in range(n_epochs):
        with tqdm.tqdm(total=len(train_loader), desc="Epoch {}".format(epoch + 1), unit='b', leave=False) as pbar:
            # Training phase
            model.train()
            running_loss = 0.
            n_correct_pred = 0
            n_samples = 0
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                batsz = inputs.size()[0]
                x_in = torch.zeros(batsz, 3, BL, BL * 2)
                y_in = torch.zeros(batsz)
                for idx in range(batsz):
                    p1, p2 = get_index(0.4)

                    img1 = subsqaureLxL(inputs[idx], p1 // L, p1 % L)
                    img2 = subsqaureLxL(inputs[idx], p2 // L, p2 % L)
                    x_in[idx, :, :, 0:BL] = img1
                    x_in[idx, :, :, BL:2*BL] = img2
                    '''
                    plt.imshow(x_in[idx].permute(1, 2, 0))
                    plt.show()
                    plt.imshow(inputs[idx].permute(1, 2, 0))
                    plt.show()
                    '''
                    if check_adjust(p1, p2):
                        y_in[idx] = 1

                pre_process(x_in)
                if is_cuda_available:
                    x_in, y_in = Variable(x_in.cuda()), Variable(y_in.cuda())
                else:
                    x_in, y_in = Variable(x_in), Variable(y_in)

                optimizer.zero_grad()
                outputs = model(x_in).view(batsz)
                loss = criterion(outputs, y_in)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x_in.size()[0]

                for idx in range(batsz):
                    if (y_in[idx].item()) == (outputs[idx] > 0.5):
                        n_correct_pred += 1

                n_samples += inputs.size()[0]
                pbar.update(1)

            loss_history.append(running_loss / n_samples)
            acc_history.append(n_correct_pred / n_samples)

            # Validation phase
            model.eval()
            running_loss = 0.
            n_correct_pred = 0
            n_samples = 0
            for i, data in enumerate(validation_loader, 0):
                inputs, _ = data
                batsz = inputs.size()[0]
                n_samples += batsz
                x_in = torch.zeros(batsz, 3, BL, BL * 2)
                y_in = torch.zeros(batsz)
                for idx in range(batsz):
                    p1, p2 = get_index(0.4)

                    img1 = subsqaureLxL(inputs[idx], p1 // L, p1 % L)
                    img2 = subsqaureLxL(inputs[idx], p2 // L, p2 % L)
                    x_in[idx, :, :, 0:BL] = img1
                    x_in[idx, :, :, BL:2 * BL] = img2
                    if check_adjust(p1, p2):
                        y_in[idx] = 1

                outputs = model(x_in).view(batsz)
                loss = criterion(outputs, y_in)
                running_loss += loss.item() * x_in.size()[0]
                for idx in range(batsz):
                    if (y_in[idx].item()) == (outputs[idx] > 0.5):
                        n_correct_pred += 1

            val_loss_history.append(running_loss / n_samples)
            val_acc_history.append(n_correct_pred / n_samples)

            # Update the progress bar.
            print("Epoch {0:03d}: loss={1:.4f}, val_loss={2:.4f}, acc={3:.2%}, val_acc={4:.2%}".format(
                epoch + 1, loss_history[-1], val_loss_history[-1], acc_history[-1], val_acc_history[-1]))
    print('Training completed')
    history = {
        'loss': loss_history,
        'val_loss': val_loss_history,
        'acc': acc_history,
        'val_acc': val_acc_history
    }
    # Save the model when requested.
    if save_file_name is not None:
        torch.save({
            'history': history,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_file_name + '.model')
    return history


mx_val = -1
perm_d = []


def dfs(idx, bcnt, bset, p1, p2, p3, matrix, prob, now_list):
    global mx_val, perm_d
    if prob < 0.01:
        return

    if bcnt == L2:
        if mx_val < prob:
            mx_val = prob
            perm_d = copy.deepcopy(now_list)

    for p4 in range(L2):
        if (bset & (1 << p4)) is 0:
            pos = prob
            if bcnt >= 3:
                pos *= matrix[idx][1][p1][p4].item()
            if bcnt % 3 is not 0:
                pos *= matrix[idx][0][p3][p4].item()

            now_list[bcnt] = p4
            dfs(idx, bcnt + 1, bset | 1 << p4, p2, p3, p4, matrix, pos, now_list)


def make_perm_from_images(model, inputs, input_perms):
    global mx_val, perm_d
    batsz = inputs.size()[0]
    matrix = torch.zeros(batsz, 2, L2, L2)
    x_in = torch.zeros(batsz, 3, BL, BL * 2)
    for p1 in range(L2):
        for p2 in range(L2):
            if p1 == p2:
                continue
            for idx in range(batsz):
                img1 = subsqaureLxL(inputs[idx], input_perms[idx][p1] // L, input_perms[idx][p1] % L)
                img2 = subsqaureLxL(inputs[idx], input_perms[idx][p2] // L, input_perms[idx][p2] % L)
                x_in[idx, :, :, 0:BL] = img1
                x_in[idx, :, :, BL:2 * BL] = img2

            outputs = model(x_in).view(batsz)
            for idx in range(batsz):
                matrix[idx][0][p1][p2] = outputs[idx]

            for idx in range(batsz):
                img1 = subsqaureLxL(inputs[idx], input_perms[idx][p1] // L, input_perms[idx][p1] % L)
                img2 = subsqaureLxL(inputs[idx], input_perms[idx][p2] // L, input_perms[idx][p2] % L)
                x_in[idx, :, :, 0:BL] = torch.transpose(img1, 1, 2)
                x_in[idx, :, :, BL:2 * BL] = torch.transpose(img2, 1, 2)

            outputs = model(x_in).view(batsz)
            for idx in range(batsz):
                matrix[idx][1][p1][p2] = outputs[idx]

    dp = np.zeros((2 ** L2, L2, L2, L2))
    trace = np.zeros((2 ** L2, L2, L2, L2))
    vis = np.zeros((2 ** L2, L2, L2, L2))

    def get_dp(idx, bcnt, bset, p1, p2, p3, vis, dp, trace, matrix):
        if vis[bset][p1][p2][p3] == 1:
            return dp[bset][p1][p2][p3]
        if bcnt == L2:
            return 1.0

        max_pos = -1
        for p4 in range(L2):
            if (bset & (1 << p4)) is 0:
                pos = 1
                if bcnt >= 3:
                    pos *= matrix[idx][1][p1][p4].item()
                if bcnt % 3 is not 0:
                    pos *= matrix[idx][0][p3][p4].item()

                val = get_dp(idx, bcnt + 1, bset | 1 << p4, p2, p3, p4, vis, dp, trace, matrix) * pos
                if max_pos < val:
                    max_pos = val
                    trace[bset][p1][p2][p3] = p4

        vis[bset][p1][p2][p3] = 1
        dp[bset][p1][p2][p3] = max_pos
        return dp[bset][p1][p2][p3]

    res_perms = []
    for idx in range(batsz):
        now_list = np.zeros(L2)
        mx_val = -1
        dfs(idx, 0, 0, 0, 0, 0, matrix, 1.0, now_list)
        if mx_val > 0:
            res_perms.append(perm_d)
        else:
            dp.fill(0)
            trace.fill(0)
            vis.fill(0)

            val = get_dp(idx, 0, 0, 0, 0, 0, vis, dp, trace, matrix)
            bset, p1, p2, p3 = 0, 0, 0, 0

            res_p = []
            for i in range(L2):
                v = int(trace[bset][p1][p2][p3])
                res_p.append(v)
                p1 = p2
                p2 = p3
                p3 = v
                bset |= 1 << v
            res_perms.append(res_p)

    return res_perms

# Test process
def test_model(model, test_loader):
    global mx_val, perm_d
    model.eval()
    n_correct_pred = 0
    n_samples = 0
    for i, data in enumerate(validation_loader, 0):
        inputs, _ = data
        batsz = inputs.size()[0]
        perms = []
        for i in range(batsz):
            perms.append(np.random.permutation(L2))

        res_perm = make_perm_from_images(model, inputs, perms)

        for idx in range(batsz):
            correct = 1
            for i in range(L2):
                if res_perm[idx][perms[idx][i]] != i:
                    correct = 0
                    break

            if correct == 0:
                expected_perm = np.zeros(L2)
                for i in range(L2):
                    expected_perm[perms[idx][i]] = i
                print('expected : ', expected_perm)
                print('result : ', res_perm[idx])
            n_correct_pred += correct
            n_samples += 1

        print("sample = {}, correct = {}".format(n_samples, n_correct_pred))
    return n_correct_pred / n_samples


parser = argparse.ArgumentParser()
parser.add_argument('epoch', type=int, help='# of epoch (1 time ~= 40s)', default=100)
parser.add_argument('lr', type=float, help='learning rate (0.01 ~ 0.0001)', default=0.001)
args = parser.parse_args()
n_epochs = args.epoch
lr = args.lr

# Create the neural network.
model = RectConvNet_3x3()
is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    model.cuda()

n_params = 0
for p in model.parameters():
    n_params += np.prod(p.size())
print('# of parameters: {}'.format(n_params))

# We use binary cross-entropy loss here.
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
save_file_name = 'jigsaw_cifar100_n{}_lr{}'.format(n_epochs, lr)
history = train_model(model, criterion, optimizer, train_loader, validation_loader,
                      n_epochs=n_epochs, save_file_name=save_file_name)

plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig(save_file_name+'_loss.png', dpi=300)
plt.figure()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig(save_file_name+'_accuracy.png', dpi=300)

# Calculate accuracy

print('Training accuracy: {}'.format(test_model(model, train_loader)))
print('Validation accuracy: {}'.format(test_model(model, validation_loader)))
print('Test accuracy: {}'.format(test_model(model, test_loader)))


# Let us try some test images.
test_data_iter = iter(test_loader)
test_images, _ = test_data_iter.next()
p_images, perms = permuteLxL(test_images)
for idx in range(p_images.size()[0]):
    for i in range(32):
        for j in range(32):
            if i >= BL * L or j >= BL * L:
                for c in range(3):
                    p_images[idx][c][i][j] = 0

# Show permuted images.
plt.figure()
imshow(torchvision.utils.make_grid(p_images))
plt.title('Inputs')
plt.savefig(save_file_name+'_shuffled_sample.png', dpi=300)

model.eval()
iperms = []
for i in range(p_images.size()[0]):
    iden = []
    for j in range(L2):
        iden.append(j)
    iperms.append(iden)
perms2 = make_perm_from_images(model, p_images, iperms)

# Show restored images.
plt.figure()
imshow(torchvision.utils.make_grid(restoreLxL(p_images, perms2)))
plt.title('Restored')
plt.savefig(save_file_name+'_restored_sample.png', dpi=300)
