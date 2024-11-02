class RegTrainer(Trainer):
    def setup(self):
        """Initial setup for datasets, model, loss, optimizer, and early stopping."""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("GPU is not available")

        # Early stopping variables
        self.patience = 10  # Number of epochs with no improvement to stop training
        self.no_improvement_epochs = 0
        self.best_val_loss = np.inf  # To track the best validation loss

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate if x == 'train' else default_collate),
                                          batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio,
                                   args.background_ratio, args.use_background, self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """Training process with early stopping"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()

            # Validation with early stopping check
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                val_mae = self.val_epoch()  # Capture the validation MAE from val_epoch

                # Early stopping logic
                if val_mae < self.best_val_loss:
                    self.best_val_loss = val_mae
                    self.no_improvement_epochs = 0
                    logging.info(f"Validation improved, resetting no improvement count.")
                else:
                    self.no_improvement_epochs += 1
                    logging.info(f"No improvement for {self.no_improvement_epochs} epochs.")
                    
                    if self.no_improvement_epochs >= self.patience:
                        logging.info("Early stopping triggered!")
                        break  # Stop training if patience exceeded

    def train_epoch(self):
        # Same as your previous train_epoch function, for processing training data.
        ...

    def val_epoch(self):
        """Validation process"""
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluation mode
        epoch_res = []
        
        # Iterate over validation data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'The batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("Saving best model with MSE {:.2f}, MAE {:.2f} at epoch {}"
                         .format(self.best_mse, self.best_mae, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

        return mae  # Return MAE for early stopping comparison
