# ASTrust

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

Trustworthiness and interpretability are inextricably linked concepts for LLMs. The more interpretable an LLM is, the more trustworthy it becomes. However, current techniques for interpreting LLMs when applied to code-related tasks largely focus on accuracy measurements, measures of how models react to change, or individual task performance instead of the fine-grained explanations needed at prediction time for greater interpretability, and hence trust. To improve upon this status quo, this paper introduces _ASTtrust_, an interpretability method for LLMs of code that generates explanations grounded in the relationship between model confidence and syntactic structures of programming languages. _ASTtrust_ explains generated code in the context of _syntax categories_ based on Abstract Syntax Trees and aids practitioners in understanding model predictions at _both_ local (individual code snippets) and global (larger datasets of code) levels. By distributing and assigning model confidence scores to well-known syntactic structures that exist within ASTs, our approach moves beyond prior techniques that perform token-level confidence mapping by offering a view of model confidence that directly aligns with programming language concepts with which developers are familiar. 

To put _ASTtrust_ into practice, we developed an automated visualization that illustrates the aggregated model confidence scores superimposed on sequence, heat-map, and graph-based visuals of syntactic structures from ASTs. We examine both the _practical benefit_ that _ASTtrust_ can provide through an empirical study on 12 popular LLMs on a curated set of GitHub repos and the _usefulness_ of _ASTtrust_ through a user study. Our findings illustrate that there is a _causal_ connection between learning error and an LLM's ability to predict different syntax categories according to _ASTtrust_~-- illustrating that our approach can be used to interpret model _effectiveness_ in the context of its syntactic categories. Finally, users generally found _ASTtrust_'s visualizations useful in understanding the trustworthiness of model predictions.

## What is Syntax-Interpretable Performance?

ASTrust is a method to interpret LLMs for code. We believe that evaluating and interpreting LLMs for code are closely intertwined. A contextualized evaluation of these models requires understanding their predictions at a detailed level, which demands some level of interpretability. However, achieving fine-grained evaluation poses a challenge because it involves establishing a reliable mapping mechanism between the model's predictions and developer-friendly syntactic concepts, aiding in interpreting the model's decisions. Therefore, to design effective and interpretable evaluation metrics for LLMs, it is crucial to establish this conceptual mapping first. To address the challenges of fine-grained evaluation, we propose a novel approach called Syntax-Grounded Explanations (ASTrust). This approach explicitly maps token predictions to code structure, offering an interpretable and detailed methodology for aggregating LLM token predictions into constructs referred to as "Syntax Categories", which are derived from Abstract Syntax Trees (ASTs).


![fig_1_approach](https://github.com/WM-SEMERU/CodeSyntaxConcept/assets/1779831/5ef2267e-e34e-4702-8b4b-aa854b8b5ad3)



The precondition to using (SIP $`\alpha`$ ) is to have a held-out testbed and
an LLM under analysis. The first step, **inference**, is to generate the
Next Token Predictions of each sample in the testbed. The second,
**evaluation**, step is to compute *Cross-Entropy Loss* and our
aggregation metric AsC-*Eval*. The third step, **explainability**,
measures the causal effect of AsC-*Eval* to Cross-Entropy.

## Structural Quantitative Design

While LLMs have seen striking advances in code generation
and other downstream SE tasks, researchers are still not able to
evaluate what aspects of code are statistically learned by
these models. We propose a new metric, AsC-*Eval*, to showcase the
statistical behavior of syntactic elements generated by LLMs. AsC-*Eval*
comprises the basic units for explainability (see Fig. below) as
Abstract Syntax Concepts (AsC), an alignment function $\delta$ that
links tokens with ASTs, and an aggregation function $\theta$ that
estimates the prediction performance of a terminal and non-terminal
nodes. We propose an explainability function $\varphi$ that relies on
the alignment function $\delta$ and the aggregation function $\theta$ to
perform the mapping from log-probabilities to developer-understandable
concepts. AsC-*Eval*: Left: Nodes are employed as *concepts*. Center:
Each token is aligned to the end nodes of the AST with an offset
function. Right: Node probabilities are estimated with an aggregation
function.
![fig_2_asc_causal](https://github.com/WM-SEMERU/CodeSyntaxConcept/assets/1779831/24f65453-0247-4f20-a34e-59dbf9903e8e)



## What is AsC-*Causal*?

AsC-*Causal* component can be used to explain and contextualize other
canonical metrics such as the *cross-entropy loss*. To achieve that, we
propose a causal inference technique to estimate the impact of the Abstract
Syntax Concepts (AsC) predictions on overall LLM performance. We can
explain the prediction performance of LLMs using AsC-*Eval* values as
treatment effects. These effects are computed from **Structural Causal
Model** (SCM), which represents our assumptions about the underlying
causal process. In our study, these assumptions take the form of the
performance of each AsC (treatments $T$), code features (confounders
$Z$), and the LLMs canonical performance (outcome $Y$). The relationship
or directionality information of these causal variables is explicitly
stated in the SCM (see Fig below). The goal of the causal analysis is to
determine the *Average Treatment Effect* (ATE) that the prediction of
*AsC* has on the Cross-Entropy after controlling the confounding
variables. In other words, we want to estimate the probability
$p(Y|do(T))$ to identify cases of *spurious correlations* (*i.e.,*
association is not causation)

![boxplot](./figures/approach/asccausal.png "Causal")

## Sysntax-Interpretable Approach

The visualization component for SIP $`\alpha`$ is a graphical explainability
technique that displays the the SIP performance values of the
terminal and non-terminal nodes for a *single* local evaluation. We take
advantage of the hierarchical structure of PLs to visually accommodate
SIP values into the AST. Fig. below illustrates how we
accommodate the SIP values for a code generation task using
*gpt-3* model. Region **1** shows a box with a prompt with an incomplete
snippet followed by a second box with generated tokens in blue. Then, in
region **2**, the resulting auto-completed snippet is processed with
AsC-*Eval* and represented as an AST. Each node has information about
the AsC-*Eval* performance after applying local aggregations $\theta$.
The nodes are color-coded. The highest aggregated values (*i.e.,* best
predictions) are displayed in shades of blue. In contrast, nodes with
the smallest values (*i.e.,* worst predictions) are displayed in shades
of red. Nodes, in region **2**, encapsulate the code tokens generated by
the LLM as presented in region **3**. We refer to tokens linearly
organized as *sequence representation*.
![fig_1_ast_tree](https://github.com/WM-SEMERU/CodeSyntaxConcept/assets/1779831/dab3e16a-6507-4a72-8314-0effb0766060)



# Replication Package

## Code & Data

Below we provide links to the SIP $`\alpha`$ data set and framework API.
The code under the folder `CodeSyntaxConcept/` is organized as
follows: - AST Generation: `loader.py`, `parser.py`, `tokenizer.py` -
Aggregation Function: `aggregator.py`, `statistics.py` - Alignment
Function: `embedding.py` - Logits (Next Token Prediction) Generator:
`extractor.py`

The API is found in this link: `github Pages` The **SyxTestbed** dataset
can be found here:

## Usage

Original empirical analysis notebooks are under the folder
`CodeSyntaxConcept/experimental_notebooks/`.

Logit extractor works with HugginFace API `CausalLMOutputWithPast`

``` python
def logit_extractor(batch, input, from_index=0):
    """
    """
    #Output is in CausalLMOutputWithPast
    CODEMODEL =  params['codemodel']
    create_folder(params['numpy_files_logits_path'])

    for idx, n in enumerate( range( from_index, len(input), batch) ):
        output = [ model( 
            input_ids = i, 
            labels = i.type(torch.LongTensor).to(device) 
            ) for i in input[n:n+batch] ] #Labels must be provided to compute loss
    
        output_logits = [ o.logits.detach().to('cpu').numpy() for o in output ]  #Logits Extraction
        output_loss = np.array([ o.loss.detach().to('cpu').numpy() for o in output ])  #Language modeling loss (for next-token prediction).

        #Saving Callbacks
        current_batch = idx + (from_index//batch)
        for jdx, o_logits in enumerate( output_logits ):
            np.save( params['numpy_files_logits_path']+ '/'+ f'logits_tensor[{jdx+n}]_batch[{current_batch}]_model[{CODEMODEL}].npy', o_logits) #Saving LOGITS
        np.save( params['numpy_files_logits_path']+ '/'+f'loss_batch[{current_batch}]_model[{CODEMODEL}].npy', output_loss) #Saving LOSS
        
        logging.info(f"Batch [{current_batch}] Completed")
        
        #Memory Liberated
        for out in output:
            del out.logits
            torch.cuda.empty_cache()
            del out.loss
            torch.cuda.empty_cache()
        for out in output_logits:
            del out
            torch.cuda.empty_cache()
        for out in output_loss:
            del out
            torch.cuda.empty_cache()
        del output
        del output_logits
        del output_loss
```

## Empirical Results

### RQ1 AsC Performance Evaluation

*To what extent do Large Language Models for code predict syntactic
structures?* The prediction of syntactic structures highly depends on
LLMs’ parameter size and fine-tuning strategy. More specifically the
largest evaluated mono language model (2B) , which was fine-tuned with
the BigPython and BigQuery datasets, obtains the highest global average
AsC-*Eval* Performace of $0.84$ with the lowest variability.

![boxplot](./figures/results/rq1/ascperformance.png "ascperformance")

- The bar plot below depicts the percentage increments of AsC-*Eval*
  values between baseline and the largest models.

![boxplot](./figures/results/rq1/increment_bars.png "ascperformance")

### RQ2 Empirical Causal Evaluation

*How do Abstract Syntax Concepts impact LLMs’ canonical prediction
performance?* We can observe that cross-entropy loss of LLMs tends to be
negatively impacted by the AsC-*Eval* values at snippet granularity. The
figure below shows how the function definition concept impacts the
cross-entropy.

![boxplot](./figures/results/rq2/output_2.7B_corr_loss.png "corr_loss_ccp")

### RQ3 User Study on AsC Visualization

*How useful is our AST evaluation method for developers in a practical
scenario?* We proposed 3 different treetments. The following reports
summarize the user study:

- Treatment 1: [A complete AST
  Representation](https://github.com/xxxxx/CodeSyntaxConcept/blob/master/reports/AST-Complete.pdf)
- Treatment 2: [A partial AST
  Representation](https://github.com/xxxxx/CodeSyntaxConcept/blob/master/reports/AST-Partial.pdf)  
- Treatment 3: [Sequence
  Representation](https://github.com/xxxxx/CodeSyntaxConcept/blob/master/reports/Sequence.pdf)

## Data Collection

Our selected LLMs were trained on datasets such as BigQuery, BigPython,
and Pile, which are based on code repositories from Github up until
2021. We assume that our tested models have already seen the common
testing datasets, so it is unfair to test code generation with trained
data. Therefore, we generate a new dataset that consists of recent
commits performed from January 1st, 2022 to January 1st, 2023. We
selected Python repositories from Github that have more than one hundred
stars. We took the commit differences, picked changed files and changed
methods, and extracted the code snippet for all new and updated methods.
Since two or more commits can affect the same method, we deleted
duplicated ones. During the process, we generated the Abstract Syntax
Tree (AST) for each method and obtained the commit message, code,
comments (if any), number of nodes, AST levels, AST errors, whitespaces,
lines of code, cyclomatic complexity, and token counts. We obtained a
total of 50971 unique snippets.

To build this extraction pipeline we: 1. Used the
[pydrill](https://pypi.org/project/pydrill/) library for git repository
mining 2. We filtered GitHub Python repositories with commits between
January 1st 2022 and January 1st 2023 and over 100 starts 3. For each
repository we extracted the diff change and extracted the changed
methods 4. We yous extracted the added and changed method, we assume
this is a new unseen code 5. We deleted duplicated methods where the
code is exactly the same and there were small changes i.e, tabular or
white spaces changes 6. We used
[tree-sitter](https://tree-sitter.github.io/tree-sitter/) to generate
the AST for each method 7. We save the code and all related features to
the output Json file
