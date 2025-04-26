# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App()

@app.cell
def _():
    import marimo as mo # Run me before anything else!
    return (mo,)

@app.cell
def _(mo):
    mo.md(
        r"""
        Embeddings
        ===
        MSOE AI Club Workshop
        ```
          _____________
         /0   /     \  \
        /  \ M A I C/  /\
        \ / *      /  / /
         \___\____/  @ /
                  \_/_/
        ```
        (Rosie is not needed!)

        Job listing dataset credit: https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description (You don't need to download this yourself)

        Run the below pip installs now so we don't have to wait for them later:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        Last year, [an article](https://foojay.io/today/indexing-all-of-wikipedia-on-a-laptop/) was published showing that it's possible to search all of Wikipedia using only the compute power of a laptop. The article itself is centered around the use of some Java library, but today we'll be looking at the underlying theory that makes something like this even possible: embeddings.

        Embeddings are an extremely useful tool in modern machine learning, allowing raw text to be transformed into numerical representations that computers can understand.
        They are also a popular interview question to test a candidate’s understanding of vector spaces, similarity metrics, and real-world applications.
        Beyond that, embeddings are incredibly common in ML, powering everything from search engines and recommendation systems to chatbots and fraud detection.
        You'll see embeddings being used everywhere if you look! Here are just some models, projects, and papers that make use of embeddings:
        - [The original transformer paper - the basis of modern LLMs](https://arxiv.org/pdf/1706.03762)
        - [RAG systems - often used to give LLMs comprehensive access to much more information than they could normally use at once](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
        - [Image generations models such as Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion)
          - Note: images are generated *in embedding space*!
        - [Audio-continuation models such as RAVE](https://github.com/acids-ircam/RAVE)
        - Modern image search makes extensive use of embeddings
        - Modern recommendation algorithms also use embeddings
        - Even some papers published by MSOE students involve the use of embeddings! Here are a few:
          - [Agent simulation with LLMs](https://arxiv.org/pdf/2409.13753)
          - [Strategy masking - a technique to control model behavior](https://arxiv.org/pdf/2501.05501)

        **What *are* Embeddings?**

        At the lowest level, an embedding is just stored as a list of numbers. This could be an embedding: `[0.1, 0.2, -0.3]`.

        This list of numbers is best interpreted as a point or direction in some very high-dimensional space that represents something. In the case of text-based models, embeddings are used to represent words and sentences.

        In practice, embeddings range from tens of dimensions to over 1000. For simplicity, let's only conceptualize things in two or three dimensions for now - that way we can actually visualize what's going on.

        The image below shows how embedded words can be thought of as directions in space. We're specifically looking at words in the phrase `some embedded text`. Each embedding point describes direction relative to the point (0,0).

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img1.png" width=1000px>

        But how do we actually interpret these directions in space as being words? The answer is that different directions in the space represent different aspects of a word -
        - one direction may encode "past tense,"
        - another may enode the idea of "running" or "to run."

        In the case above, the embedding of the word "ran" may point in the average of the directions encoding "past tense," and "to run."

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img2.png" width=600px>

        This topic naturally leads into another important point: embeddings *closer* in embedding space are also *closer* in meaning. The word "ran" will be closer to "walked" than to "stapler." This is the case, because words with increasingly different meanings are, *by definition,* pointing in increasingly different directions to encode those meanings.

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img3.png" width=600px>

        NOTE: we'll watch this during the workshop:

        [Here is a one-minute video that illustrates this concept using real-world embeddings.](https://www.youtube.com/watch?v=FJtFZwbvkI4)

        **How is it possible for there to be directions dedicated to ideas as specific as "Italian-ness" and "WWII Axis leaders?"**

        One might expect that the directions in embedding space would represent more general concepts.

        If directions can be allocated to specific ideas like "WWII Axis leaders," how are there enough directions left to represent everything else, from "60s British pop bands" to "computer keyboard layouts"?!?!

        In two or three dimensions, it's *not* really possible to have directions this specific. But, remember that text embeddings are typically 10s to 1000s of dimensions.  

        As the number of dimensions grows, the number of possible points and directions in a space grows MUCH more quickly. More directions means more unique aspects of a word can be encoded.

        Let's work with the constraint that points of different meanings must be one unit apart. This is somewhat arbitrary, but it is true that there is a "minumum" distance between two points before they mean the same thing. Let's also say that we only allow points in the range 0 to 1. This is also somewhat arbitrary, but machine learning models often try to keep numbers from getting too big to prevent numbers from going to infinity. With these constraints, we can only fit two points in one dimension:

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img4.png" width=600px>

        These two points (or two directions relative to a centerpoint) probably can't encode much information. Through the lens of our prior examples, this one-dimensional space could only encode two opposite meanings, E.g., "run" and "walk."

        Now what if we extrude ourselves into the second dimension with the same constraints? 

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img5.png" width=600px>

        We now have *four* points (or four directions). This means we can differentiate more words. For instance, "run", "stroll", "walk", and "jog" could have unique directions.

        And if we went to three dimensions we'd have eight points - imagine extruding the four points of this square into a cube.

        In general, our constraints will allow N dimensions to encode $2^N$ unique directions.

        - With 10 dimensions, you have over 1000 directions.
        - 20 dimensions gets us over 1 million directions.
        - And at 1000 dimensions, we have **more possible unique directions than atoms in the observable universe,** each of which can be interpolated between to embed specific words or sentences!

        The act of adding just *one* dimension EXPONENTIALLY increases how many things we can fit in the space! So think about adding a dimension to a 3D space... *1000 times*.

        <img src="https://www.i2tutorials.com/wp-content/media/2019/09/Curse-of-Dimensionality-i2tutorials.png" width=1000px>

        Although we can't *see* the directions encoding things like "WWII Axis leaders," there is no doubt that these directions are able to exist.

        **Who decided that there should be directions for these particular ideas?**

        These directions are not something humans designed directly. Instead, these directions *emerge* from the process of training the model.

        The model learns from a huge amount of text and starts to recognize patterns, like which words tend to appear in similar contexts.

        As it processes more and more language, the model "figures out" what sort of information it should store in the directions of an embedding space - even though no one explicitly programmed it to do that!

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        **That seems neat. How can I use embeddings?**

        Let's set things up!

        It's really easy to get started with embeddings. You can even run small embedding models on your laptop!

        We'll be using `sentence-transformers` to run [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - a model that embeds sentences into 384 dimensions.
        """
    )
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2") # Our model of choice is supplied here. You can find many more on huggingface: https://huggingface.co/models?sort=trending&search=embed
    embedding = model.encode("This is an embedded text example.")
    return embedding, model, np


@app.cell
def _(mo):
    mo.md(r"""Our embeddings are just lists of numbers stored as Numpy arrays. Numpy is just a library that makes it easier to manipulate arrays.""")
    return


@app.cell
def _(embedding):
    print(embedding)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        Now that we (hopefully) have a working embedding model, let's put it to use.

        But before that, we need to understand how to measure "distance" in embedding space.

        **Q: How would you normally measure distance between points in space?**

        **A: I would use the [Pythagorean theorem to find the Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance).**

        That's right! Except we don't use Euclidean distance for embeddings.

        Well, you *could* use Euclidean distance for embeddings, but we use a different distance metric to take advantage of the fact that embeddings are directions.

        There is a metric you can compute between two vectors (two directions) called the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of the two vectors. This number is simply the cosine of the angle between the vectors.

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img6.png" width=1000px>

        If an angle $\theta$ between two vectors is small and close to zero, then the cosine of that value will be close to $\cos(0) = 1$. The maximum cosine similarity is $1$.

        As the angle increases, the cosine similarity decreases. When the angle passes ninety degrees, the cosine similarity goes negative. Two opposite vectors have a cosine similarity of $-1$. **Higher cosine similarity means two vectors are more similar.**

        The cosine similarity is so useful not only because it can tell us the similarity between vectors, but *also* because it's really easy to calculate. Simply multiply each number in a vector with each number in the same position in another, and then take the sum:
        $$\text{CosineSim}([1,2], [3,4]) = 1\times2\ +\ 2\times4.$$

        This quantity is also called the [dot product](https://en.wikipedia.org/wiki/Dot_product), and we'll be computing it via `np.dot`.

        DISCLAIMER: technically, the dot product only equals the cosine similarity when the vectors are *normalized* (have a magnitude of 1), but embedding vectors are usually normalized.

        There is a separate metric from cosine similarity called cosine *distance.* It is computed via $1 - \text{CosineSim}$. Unlike cosine similarity which tells you "how similar" two vectors are, cosine distance acts more like Euclidean distance in the sense that higher numbers mean "more different" rather than "more similar." The cosine distance ranges from $0$ (for two equal vectors) to $2$ (for two opposite vectors).

        If you *did* use Euclidean distance on embeddings, your distances would be similar to those from using cosine distance. The reason that cosine distance is still preferred is computational - the Euclidean distance requires a square root to compute while cosine distance doesn't.

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        Let's first try the example from the previously linked 3Blue1Brown video.

        Recall that you can take the embedding for "Uncle," subtract the embedding for "Man," and add the embedding for "Woman." Doing so gets you a new embedding very close to "aunt."

        The embedding of "uncle" is already close to "aunt" because they share meaning as familial roles. However, by subtracting "man" and adding "woman," we are shifting "uncle" toward its female counterpart, "aunt," in the embedding space.
        """
    )
    return


@app.cell
def _(model, np):
    emb_uncle = model.encode("uncle")
    emb_aunt = model.encode("aunt")
    emb_man = model.encode("man")
    emb_woman = model.encode("woman")

    sim1 = np.dot(emb_uncle, emb_aunt) # We are using np.dot to evaluate the cosine similarity
    sim2 = np.dot(emb_uncle - emb_man + emb_woman, emb_aunt)

    print('CosSim:\t\t\t', sim1)
    print('CosSim after transform:\t', sim2) # Higher similarity! Remember that a larger number means more similar.
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **If you're done and still waiting for the workshop to continue:**

        Feel free to use different words or phrases in the example above. Does the "Hitler - Germany + Italy = Mussolini" example work? What about something like "Milwaukee - Wisconsin + Illinois = Chicago"?

        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        **How can we actually visualize these things if they're in more than 3 dimensions?**

        If we want to look at a single embedding, there are a few approaches we can take:

        - Bar graph. The x axis represents the embedding dimension, the y axis represents the dimension value.
          - This is best if you want to understand the distribution of magnitude of values within an embedding.
        - Image. The dimensions are reshaped to form a rectangle (in our case: 384 -> 16 by 24), and then each value is used to determine a pixel's brightness.
          - This is very useful if you want to see a "heatmap" of embedding space. This approach can be used to understand which dimensions are "lighting up" under certain contexts. An example of this approach can be seen in [this video about controlling LLM behavior](https://www.youtube.com/watch?v=UGO_Ehywuxc) which visualizes embeddings as images.
          - This approach can also be useful for image-embedding models. With image embeddings, certain dimensions often "light up" when certain visual features are present.

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        Let's look at some embeddings!
        """
    )
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def _(model, plt):
    NUM_BARS = 30 # set to some big number (like 1000) to see all dimensions

    emb_to_viz = model.encode("Milwaukee")

    plt.bar(range(emb_to_viz[:NUM_BARS].size), emb_to_viz[:NUM_BARS])
    plt.show()

    # Two embeddings in one plot

    emb_to_viz1 = model.encode("Milwaukee")
    emb_to_viz2 = model.encode("Chicago")

    NUM_BARS = 30
    plt.bar(range(emb_to_viz1[:NUM_BARS].size), emb_to_viz1[:NUM_BARS])
    plt.bar(range(emb_to_viz2[:NUM_BARS].size), emb_to_viz2[:NUM_BARS])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""Here is the same embedding space shown as an image.""")
    return


@app.cell
def _(model, plt):
    emb_to_viz_1 = model.encode('Milwaukee')
    plt.imshow(emb_to_viz_1.reshape(16, 24))
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""Let's multiply the dimensions for a bunch of cities and see what we get.""")
    return


@app.cell
def _(model, np, plt):
    city_emb = model.encode('Milwaukee')
    city_emb = city_emb * model.encode('London')
    city_emb = city_emb * model.encode('Hong Kong')
    city_emb = city_emb * model.encode('Melbourne')
    city_emb = city_emb * model.encode('Moscow')
    city_emb = city_emb * model.encode('Montreal')
    city_emb = city_emb * model.encode('Cairo')
    city_emb = city_emb * model.encode('Montevideo')
    city_emb = city_emb * model.encode('Toronto')
    city_emb = city_emb * model.encode('Berlin')
    city_emb = np.abs(city_emb) ** (1 / 10)
    plt.imshow(city_emb.reshape(16, 24))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        You'll notice that one or two dimensions become realy prominent, and that others remain fairly present.

        **Do these dimensions play a part in encoding information about geographical location?**

        Kind of, but there's something important to note here. Take a look at what happens if you embed the word "stapler." That single component will still look quite bright, but "stapler" isn't a place.

        So is this component just "always on" for any input? Not necessarily, try embedding "AI Club" and you'll see it go darker. What's going on here?

        Individual dimensions are very difficult to interpret, because meaning comes from directions - I.e., *combinations* of dimensions. A single large value can only be interpreted in the context of other values.

        In the case of our "city" embedding, it would be more accurate to say that the most prominent dimensions encode geographical location *in the context of other dimensions being their values, even if those values are smaller.*

        We will soon take a look at a better method for interpreting embeddings: PCA. For now, let's embed some real data...

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        **Let's put this embedding knowledge to use!**

        Since embeddings are a popular topic for interview questions, let's use embeddings to search a dataset of job listings.

        We will be embedding the descriptions of job listings. Using these embeddings, we will then be able to *search* all of the job listings by description!
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    # Did you know that you can read CSVs directly from a URL?
    jobs_df = pd.read_csv('https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/job_title_des.csv')
    return (jobs_df,)


@app.cell
def _(jobs_df):
    jobs_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Here is our dataset with job titles and job descriptions.

        Below we are taking the descriptions and putting them all through our embedding model.
        """
    )
    return


@app.cell
def _(jobs_df, model):
    N_JOBS = 500 # Embedding ALL jobs would take ~1 minute on a CPU and on Rosie this would be instant. But for time, we're doing 500 jobs.

    # Let's also save the job titles for later
    job_titles = jobs_df['Job Title'][:N_JOBS]
    job_descs = jobs_df['Job Description'][:N_JOBS]

    job_embs = model.encode(job_descs.tolist()[:500])
    return job_descs, job_embs, job_titles


@app.cell
def _(mo):
    mo.md(
        r"""
        We are using a trick to compute all the similarities in one operation (rather than using a for-loop).

        Instead of iterating over every description embedding, Numpy uses the syntax below as a shorthand for "evaluate the dot product between the query and ALL descriptions."

        ```python
        np.dot(query, job_embs[0]) # Cosine similarity between query and first job.
        job_embs[0].dot(query) # Alternate syntax
        job_embs.dot(query) # Without selecting a specific job embedding, we broadcast our singular query across ALL jobs to compute ALL similarities at once.
        ```

        Remember that the dot product is just the cosine similarity in our case. It is measuring how similar two embeddings are.
        """
    )
    return


@app.cell
def _(job_embs, model, np):
    query = model.encode("I need some Python for AI") # Search for jobs by description by changing this!
    N = 2 # get the Nth most similar job in addition to the most similar job - change this to see less similar results.

    # --- calculate similarities ---

    similarities = job_embs.dot(query) # Cosine similarities between query and ALL jobs.

    # --- find most similar jobs ---

    most_similar = np.argmax(similarities) # Index of the most similar job.
    nth_most_similar = np.argsort(similarities)[-N] # Index of the Nth most similar job.
    return (most_similar,)


@app.cell
def _(mo):
    mo.md(r"""Let's take a look at our results:""")
    return


@app.cell
def _(job_descs, job_titles, most_similar):
    print(
        f"Most similar job title: {job_titles[most_similar]}"
        f"\n\nMost similar job description:\n{job_descs[most_similar][1000:]} ..."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ... or keep going if you want to work ahead.

        ---

        You now have the knowledge to perform an embedding-powered search across any textual dataset!

        **To wrap things up, let's look at a way to visualize ALL job descriptions' embeddings at once: dimensionality reduction.**

        We're using an embedding model trained on huge amounts of text spanning many different uses of English. However, our current task only spans the language needed in job descriptions.

        So If you think about it, we aren't really making full use of our embedding space; there is some redundancy. Does this mean there could there be a way to do the same thing with less dimensions? Yes it does.

        In general, if your task doesn't span the entirety of natural language, you can perform *dimensionality reduction*! There are [many ways to reduce the number of dimensions](https://en.wikipedia.org/wiki/Dimensionality_reduction), but we'll focus on the relatively simple (yet powerful) method referred to as PCA.

        **What's a PCA?**

        PCA stands for "[principle component analysis.](https://en.wikipedia.org/wiki/Principal_component_analysis)" It's a technique that finds which directions in a high-dimensional space are the best at differentiating points in said space.

        For the sake of example, let's say we embedded our job descriptions into two dimensions and they looked like this:

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img7.png" width=400px>

        Looking at this, it seems we don't really *need* two dimensions to describe all of our jobs. We can just use a single number to say how far they are along a line. In other words: **one dimension explains 100% of the variance in our data.**

        More realistically, our data will be a bit more noisy:

        <img src="https://raw.githubusercontent.com/MSOE-AI-Club/workshops/refs/heads/main/Embeddings/img8.png" width=400px>

        But it could still be the case that the majority of variance is attributable to a single direction. In this case, we may perform PCA and find that one dimension explains 90% of variance while the other explains 10%.

        In general, PCA asks "what direction most strongly differentiates some datapoints," and then it asks "what directions is *second best* at differentiating points" and so on.

        This process repeats until you have some desired number of reduced dimensions. We will be using PCA to reduce our space from 384 dimensions down to 2.

        ---

        <span style="color:#55ff55;font-weight:bold;font-size:1.5rem;">
            GO
        </span>

        Let's use PCA on our job listing embeddings.
        """
    )
    return


@app.cell
def _():
    from sklearn.decomposition import PCA # We are using the PCA implementation from scikit-learn
    return (PCA,)


@app.cell
def _(PCA, job_embs):
    # This code does the PCA process on our job embeddings. We are reducing our embeddings down to two dimensions
    pca = PCA(n_components=2)
    job_embs_reduced = pca.fit_transform(job_embs)
    return job_embs_reduced, pca


@app.cell
def _(job_embs_reduced):
    job_embs_reduced[:10] # the first ten reduced embeddings - they're just points in 2D space!
    return


@app.cell
def _(job_embs_reduced, plt):
    plt.scatter(job_embs_reduced[:,0], job_embs_reduced[:,1])
    plt.axhline(y=0.2, color='red', linestyle='-')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **What types of jobs are clustering over y=0.2?**

        It looks like there are two clusters of points: one larger cluster below $y=0.2$ and another smaller one above $y=0.2$. Let's isolate points in these clusters and see what their job titles are.
        """
    )
    return


@app.cell
def _(job_embs_reduced, job_titles):
    print('Points in the upper cluster:')
    print(job_titles[job_embs_reduced[:,1]>0.2])

    print('\nPoints in the lower cluster:')
    print(job_titles[job_embs_reduced[:,1]<0.2])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        "iOS" and "Flutter" are both terms related to development in the Apple ecosystem. It seems we've identified a cluster related to Apple-related job positions!

        Recall that we can determine how "good" our reduced embeddings are using the "explained variance" metric. Let's investigate this for our data.
        """
    )
    return


@app.cell
def _(pca):
    print(pca.explained_variance_ratio_) # Ratio of variance explained per dim.
    print(pca.explained_variance_ratio_.sum()) # Ratio of varience explained across ALL 2 dims.
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        It seems that our 2 dimensions explain around 15% of the variance in our data. This means we aren't getting the full picture - 85% of the data remains unexplained by our 2-dimensional space.

        However, this doesn't mean our results aren't useful. Another way to think about this is that 14% of the data can be explained by only 0.5% the number of dimensions.

        If you want to go deeper, there are other dimenstionality reduction methods to try. A particularly interesting one is [umap](https://umap-learn.readthedocs.io/en/latest/) - it's like PCA, but it allows for non-linear transformations.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <span style="color:#ff5555;font-weight:bold;font-size:1.5rem;">
            STOP
        </span>

        ---

        Where else can embeddings be used?

        - Images ([Google lens](https://en.wikipedia.org/wiki/Google_Lens), [DinoV2](https://dinov2.metademolab.com/), and more)
        - Audio ([Encodec](https://github.com/facebookresearch/encodec), [RAVE](https://github.com/acids-ircam/RAVE/tree/master))
        - [AI Interpretability](ttps://www.youtube.com/watch?v=UGO_Ehywuxc)
        - Classification (you can classify [text](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) and anything else that can be embedded)
        - Generative AI ([stable diffusion generates images in embedding space (aka "latent space")](https://keras.io/examples/generative/random_walks_with_stable_diffusion_3/))
        - **Many, many, many more.**

        ### The big takeaway:

        Embeddings are so useful, because they represent abstract real-world things in a way that computers can understand.

        If a problem requires an understanding of some domain, then embeddings are the perfect tool.
        """
    )
    return


if __name__ == "__main__":
    app.run()
