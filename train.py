# ... （你的其他代码）

class ImageExtension():
    # ... （你的其他代码）

    def encode_prompt(self, folder_names: List[str]):
        # 使用文件夹名字创建文本输入
        texts = ["this is a place of " + folder_name for folder_name in folder_names]
        text_input_ids = [self.tokenize_prompt(self.tokenizer, text) for text in texts]

        prompt_embeds = []
        for text_ids in text_input_ids:
            prompt_embed = self.text_encoder(
                text_ids.to(self.text_encoder.device),
                output_hidden_states=True,
            )
            encoder_hidden_states = prompt_embed.hidden_states[-2]
            prompt_embed = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
            prompt_embeds.append(prompt_embed)

        return torch.stack(prompt_embeds)

    # ... （你的其他代码）

def main():
    # ... （你的其他代码）

if __name__ == "__main__":
    # 初始化你的 ImageExtension 对象
    image_extension = ImageExtension()

    # 设置一些超参数
    num_classes = 365  # Places365 数据集的类别数
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    # 准备数据集
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(root='E:/data/data_256', transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root='E:/data/val_256', transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(root='E:/data/test_256', transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载预训练的 ResNet 模型
    pretrained_resnet = models.resnet18(pretrained=True)

    # 创建微调模型
    model = FineTunedModel(pretrained_resnet, num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # 获取文件夹名字（类别）
            folder_names = [train_dataset.classes[label] for label in labels]

            # 将文件夹名字作为文本输入
            prompt_embeds = image_extension.encode_prompt(folder_names)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型性能
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)

                # 获取文件夹名字（类别）
                folder_names = [val_dataset.classes[label] for label in labels]

                # 将文件夹名字作为文本输入
                prompt_embeds = image_extension.encode_prompt(folder_names)

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}')

    # 保存微调后的模型
    torch.save(model.state_dict(), 'path/to/save/model.pth')

    # ... （其他可能的操作）

    main()
