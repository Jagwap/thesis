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

    def format_inputs( self,dataset,mask_train=lambda x:x.select(range(2000)),mask_eval = lambda x: x.select(range(200)),
                      format_sample = format_sample,mask_train_args = {}, mask_eval_args = {}, **args):
        formatted_dataset = dataset.map(format_sample)
        train = mask_train(formatted_dataset["train"],**mask_train_args)
        tokenized_train = train.map(self.tokenize_batch, batched=True)
        tokenized_train .set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

        evaluate = mask_eval(formatted_dataset["validation"],**mask_eval_args)
        tokenized_eval = evaluate.map(self.tokenize_batch, batched=True)
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

        return tokenized_train, tokenized_eval


    def set_dataset(self, dataset, batch_size = 1,seed = 42, args = {}):
        train,evaluate = self.format_inputs(dataset = dataset,**args)

        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,worker_init_fn=seed_worker,generator=g)
        self.eval_loader = DataLoader(evaluate, batch_size=batch_size,shuffle=False, num_workers=2, pin_memory=True,worker_init_fn=seed_worker,generator=g)

    def train(self, epochs = 3,update = lambda batch,step,optimizer: None, update_args = {}):


        for epoch in range(epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in loop:
                #batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)


                update(batch, epoch * len(self.train_loader) + loop.n,self.optimizer, **update_args)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        """
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")
        """
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

    



    def evaluate_metric_wise(self, task="squad"):
        """
        Full evaluate that includes a SQuAD branch for causal / generative LMs.
        Call `evaluate("squad")` to run SQuAD-style metrics on a causal LM by generation.
        """
        self.model.eval()
        metrics = {}

        if task == "squad":
            squad_metric = evaluate.load("squad")

            PROMPT_TEMPLATE = (
                "Q: {question}\nContext: {context}\nA: {answer}"
            )

            preds_for_metric = []
            refs_for_metric = []

            gen_kwargs = {
                "max_new_tokens": 64,
                "num_beams": 4,
                "do_sample": False,
                "early_stopping": True,
            }

            with torch.no_grad():
                for batch_i, batch in enumerate(self.eval_loader):
                    if "context" in batch and "question" in batch:
                        contexts = batch["context"]
                        questions = batch["question"]
                    elif "contexts" in batch and "questions" in batch:
                        contexts = batch["contexts"]
                        questions = batch["questions"]
                    else:
                        raise RuntimeError("Batch must provide raw 'context' and 'question' strings for generative SQuAD evaluation.")

                    if isinstance(contexts, str):
                        contexts = [contexts]
                    if isinstance(questions, str):
                        questions = [questions]

                    batch_size = len(contexts)

                    prompts = [PROMPT_TEMPLATE.format(context=contexts[i], question=questions[i]) for i in range(batch_size)]

                    enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                    generated = self.model.generate(
                        **enc,
                        **gen_kwargs
                    )

                    decoded_all = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    prompt_decoded = self.tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    for i in range(batch_size):
                        full = decoded_all[i].strip()
                        pref = prompt_decoded[i].strip()

                        if full.startswith(pref):
                            pred_text = full[len(pref):].strip()
                        else:
                            if "Answer:" in full:
                                pred_text = full.split("Answer:", 1)[1].strip()
                            else:
                                parts = full.split("\n")
                                pred_text = parts[1].strip() if len(parts) > 1 else full

                        pred_text = clean_generated_text(pred_text)

                        if "id" in batch:
                            example_id = batch["id"][i] if not isinstance(batch["id"], torch.Tensor) else str(batch["id"][i].item())
                        elif "ids" in batch:
                            example_id = batch["ids"][i] if not isinstance(batch["ids"], torch.Tensor) else str(batch["ids"][i].item())
                        else:
                            example_id = f"eval-{batch_i}-{i}"

                        preds_for_metric.append({"id": str(example_id), "prediction_text": pred_text})

                        answers_list = []
                        if "answers" in batch:
                            ans = batch["answers"]
                            if isinstance(ans, dict) and isinstance(ans.get("text", None), (list, tuple)):
                                candidate = ans["text"][i]
                                if isinstance(candidate, (list, tuple)):
                                    answers_list = candidate
                                else:
                                    answers_list = [candidate]
                            elif isinstance(ans, (list, tuple)):
                                example_ans = ans[i]
                                if isinstance(example_ans, dict) and "text" in example_ans:
                                    if isinstance(example_ans["text"], (list, tuple)):
                                        answers_list = example_ans["text"]
                                    else:
                                        answers_list = [example_ans["text"]]
                        elif "answer_texts" in batch:
                            candidate = batch["answer_texts"][i]
                            answers_list = candidate if isinstance(candidate, (list, tuple)) else [candidate]
                        else:
                            answers_list = [""]
                        refs_for_metric.append({"id": str(example_id), "answers": {"text": answers_list, "answer_start": []}})

            if len(preds_for_metric) == 0:
                raise RuntimeError("No predictions collected for SQuAD evaluation. Check batch keys and tokenizer/model compatibility.")

            squad_res = squad_metric.compute(predictions=preds_for_metric, references=refs_for_metric)
            metrics.update({"exact_match": squad_res.get("exact"), "f1": squad_res.get("f1")})

            print("SQuAD (generative) evaluation:")
            print(metrics['exact_match'])


            return metrics

        else:
            raise NotImplementedError("Other tasks not implemented in this snippet.")

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