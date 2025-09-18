from thesis.helpers import *

class ModelRunner:
    def __init__(self, model_name = "gpt2", tokenizer_name = "gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.to_device()

    def to_device(self, device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model.to(self.device)

    def set_optimizer(self, optimizer = torch.optim.AdamW, LR = 5e-5):
        self.optimizer = optimizer(self.model.parameters(), lr=LR)


    
    def tokenize_batch(self, example): 
      encoding = self.tokenizer( example["text"], truncation=True, padding="max_length", max_length=128) 
      encoding["labels"] = encoding["input_ids"][:] 
      return encoding

    def format_inputs( self,dataset,mask_train=lambda x:x.select(range(2000)),mask_eval = lambda x: x.select(range(200)),format_sample = format_sample):
        formatted_dataset = dataset.map(format_sample)
        tokenized = formatted_dataset.map(self.tokenize_batch, batched=True)


        train = mask_train(tokenized["train"])
        train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

        evaluate = mask_eval(tokenized["validation"])
        evaluate.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return train, evaluate


    def set_dataset(self, dataset, batch_size = 1,seed = 42, args = {}):
        train,evaluate = self.format_inputs(dataset = dataset,**args)

        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,worker_init_fn=seed_worker,generator=g)
        self.eval_loader = DataLoader(evaluate, batch_size=batch_size,shuffle=False, num_workers=2, pin_memory=True,worker_init_fn=seed_worker,generator=g)

    def train(self, epochs = 3,update = lambda batch,step,optimizer: None):
        num_epochs = epochs
        num_training_steps = num_epochs * len(self.train_loader)
        warmup_steps = int(0.06 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                update(batch, epoch * len(self.train_loader) + loop.n,self.optimizer)

                self.optimizer.zero_grad() 
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss

                scaler.scale(loss).backward()

                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                loop.set_postfix(loss=loss.item())

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")

    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Average evaluation loss: {avg_loss:.4f}")
        perplexity = math.exp(min(avg_loss, 100))

        print(f"Perplexity: {perplexity:.4f}")

    
    
    def evaluate_on_itself(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(shift_labels.size())

                masked_loss = loss_per_token * shift_attention_mask
                total_loss += masked_loss.sum().item()
                total_tokens += shift_attention_mask.sum().item()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100))
        print(f"Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        return avg_loss, perplexity