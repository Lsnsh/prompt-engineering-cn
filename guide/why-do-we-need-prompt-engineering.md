# Why do we need prompt engineering?

Up above, we used an analogy of prompts as the “source code” that a language model “interprets”. **Prompt engineering is the art of writing prompts to get the language model to do what we want it to do** – just like software engineering is the art of writing source code to get computers to do what we want them to do.

When writing good prompts, you have to account for the idiosyncrasies of the model(s) you’re working with. The strategies will vary with the complexity of the tasks. You’ll have to come up with mechanisms to constrain the model to achieve reliable results, incorporate dynamic data that the model can’t be trained on, account for limitations in the model’s training data, design around context limits, and many other dimensions.

There’s an old adage that computers will only do what you tell them to do. **Throw that advice out the window**. Prompt engineering inverts this wisdom. It’s like programming in natural language against a non-deterministic computer that will do anything that you haven’t guided it away from doing. 

There are two broad buckets that prompt engineering approaches fall into.

## Give a Bot a Fish

The “give a bot a fish” bucket is for scenarios when you can explicitly give the bot, in the hidden context, all of the information it needs to do whatever task is requested of it.

For example, if a user loaded up their dashboard and we wanted to show them a quick little friendly message about what task items they have outstanding, we could get the bot to summarize it as

> You have 4 receipts/memos to upload. The most recent is from Target on March 5th, and the oldest is from Blink Fitness on January 17th. Thanks for staying on top of your expenses!

by providing a list of the entire inbox and any other user context we’d like it to have.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233465165-e0c6b266-b347-4128-8eaa-73974e852e45.png" title="GPT-3 summarizing a task inbox.">
</p>

Similarly, if you were helping a user book a trip, you could:

- Ask the user their dates and destination.
- Behind the scenes, search for flights and hotels.
- Embed the flight and hotel search results in the hidden context.
- Also embed the company’s travel policy in the hidden context.

And then the bot will have real-time travel information + constraints that it
can use to answer questions for the user. Here’s an example of the bot
recommending options, and the user asking it to refine them:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233465425-9e06320c-b6d9-40ef-b5a4-c556861c1328.png" title="GPT-4 helping a user book a trip.">
</p>
<details>

  <summary>(Full prompt)</summary>

```
Brex is a platform for managing business expenses. 

The following is a travel expense policy on Brex:

- Airline highest fare class for flights under 6 hours is economy.
- Airline highest fare class for flights over 6 hours is premium economy.
- Car rentals must have an average daily rate of $75 or under.
- Lodging must have an average nightly rate of $400 or under.
- Lodging must be rated 4 stars or higher.
- Meals from restaurants, food delivery, grocery, bars & nightlife must be under $75
- All other expenses must be under $5,000.
- Reimbursements require review.

The hotel options are:
| Hotel Name | Price | Reviews |
| --- | --- | --- |
| Hilton Financial District | $109/night | 3.9 stars |
| Hotel VIA | $131/night | 4.4 stars |
| Hyatt Place San Francisco | $186/night | 4.2 stars |
| Hotel Zephyr | $119/night | 4.1 stars review |

The flight options are:
| Airline | Flight Time | Duration | Number of Stops | Class | Price |
| --- | --- | --- | --- | --- | --- |
| United | 5:30am-7:37am | 2hr 7 min | Nonstop | Economy | $248 |
| Delta | 1:20pm-3:36pm | 2hr 16 min | Nonstop | Economy | $248 |
| Alaska | 9:50pm-11:58pm | 2hr 8 min | Nonstop | Premium | $512 |

An employee is booking travel to San Francisco for February 20th to February 25th.

Recommend a hotel and flight that are in policy. Keep the recommendation concise, no longer than a sentence or two, but include pleasantries as though you are a friendly colleague helping me out:
```
 
</details>

This is the same approach that products like Microsoft Bing use to incorporate dynamic data. When you chat with Bing, it asks the bot to generate three search queries. Then they run three web searches and include the summarized results in the hidden context for the bot to use.

Summarizing this section, the trick to making a good experience is to change the context dynamically in response to whatever the user is trying to do.

> 🧙‍♂️ Giving a bot a fish is the most reliable way to ensure the bot gets a fish. You will get the most consistent and reliable results with this strategy. **Use this whenever you can.**

### Semantic Search

If you just need the bot to know a little more about the world, [a common approach is to perform a semantic search](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

A semantic search is oriented around a document embedding – which you can think of as a fixed-length array[^5] of numbers, where each number represents some aspect of the document (e.g. if it’s a science document, maybe the  843rd number is large, but if it’s an art document the 1,115th number is large – this is overly simplistic, but conveys the idea).[^6]

In addition to computing an embedding for a document, you can also compute an embedding for a user query using the same function. If the user asks “Why is the sky blue?” – you compute the embedding of that question and, in theory, this embedding will be more similar to embeddings of documents that mention the sky than embeddings that don’t talk about the sky.

To find documents related to the user query, you compute the embedding and then find the top-N documents that have the most similar embedding. Then we place these documents (or summaries of these documents) in the hidden context for the bot to reference.

Notably, sometimes user queries are so short that the embedding isn’t particularly valuable. There is a clever technique described in [a paper published in December 2022](https://arxiv.org/pdf/2212.10496.pdf) called a “Hypothetical Document Embedding” or HyDE. Using this technique, you ask the model to generate a hypothetical document in response to the user’s query, and then compute the embedding for this generated document. The model  fabricates a document out of thin air – but the approach works!

The HyDE technique uses more calls to the model, but for many use cases has notable boosts in results.

[^5]: Usually referred to as a vector.
[^6]: The vector features are learned automatically, and the specific values aren’t directly interpretable by a human without some effort.

## Teach a Bot to Fish

Sometimes you’ll want the bot to have the capability to perform actions on the user’s behalf, like adding a memo to a receipt or plotting a chart. Or perhaps we want it to retrieve data in more nuanced ways than semantic search would allow for, like retrieving the past 90 days of expenses.

In these scenarios, we need to teach the bot how to fish.

### Command Grammars

We can give the bot a list of commands for our system to interpret, along with descriptions and examples for the commands, and then have it produce programs composed of those commands.

There are many caveats to consider when going with this approach. With complex command grammars, the bot will tend to hallucinate commands or arguments that could plausibly exist, but don’t actually. The art to getting this right is enumerating commands that have relatively high levels of abstraction, while giving the bot sufficient flexibility to compose them in novel and useful ways.

For example, giving the bot a `plot-the-last-90-days-of-expenses` command is not particularly flexible or composable in what the bot can do with it. Similarly, a `draw-pixel-at-x-y [x] [y] [rgb]` command would be far too low-level. But giving the bot a `plot-expenses` and `list-expenses` command provides some good primitives that the bot has some flexibility with.

In an example below, we use this list of commands:

| Command | Arguments | Description |
| --- | --- | --- |
| list-expenses | budget | Returns a list of expenses for a given budget |
| converse | message | A message to show to the user |
| plot-expenses | expenses[] | Plots a list of expenses |
| get-budget-by-name | budget_name | Retrieves a budget by name |
| list-budgets | | Returns a list of budgets the user has access to |
| add-memo | inbox_item_id, memo message | Adds a memo to the provided inbox item |

We provide this table to the model in Markdown format, which the language model handles incredibly well – presumably because OpenAI trains heavily on data from GitHub.

In this example below, we ask the model to output the commands in [reverse polish notation](https://en.wikipedia.org/wiki/Reverse_Polish_notation)[^7].

[^7]: The model handles the simplicity of RPN astoundingly well.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233505150-aef4409c-03ba-4669-95d7-6c48f3c2c3ea.png" title="A bot happily generating commands to run in response to user queries.">
</p>

> 🧠 There are some interesting subtle things going on in that example, beyond just command generation. When we ask it to add a memo to the “shake shack” expense, the model knows that the command `add-memo` takes an expense ID. But we never tell it the expense ID, so it looks up “Shake Shack” in the table of expenses we provided it, then grabs the ID from the corresponding ID column, and then uses that as an argument to `add-memo`.

Getting command grammars working reliably in complex situations can be tricky. The best levers we have here are to provide lots of descriptions, and as **many examples** of usage as we can. Large language models are [few-shot learners](https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing)), meaning that they can learn a new task by being provided just a few examples. In general, the more examples you provide the better off you’ll be – but that also eats into your token budget, so it’s a balance.

Here’s a more complex example, with the output specified in JSON instead of RPN. And we use Typescript to define the return types of commands.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233505696-fc440931-9baf-4d06-80e7-54801532d63f.png" title="A bot happily generating commands to run in response to user queries.">
</p>

<details>

  <summary>(Full prompt)</summary>
  
~~~
You are a financial assistant working at Brex, but you are also an expert programmer.

I am a customer of Brex.

You are to answer my questions by composing a series of commands.

The output types are:

```typescript
type LinkedAccount = {
    id: string,
    bank_details: {
        name: string,
        type: string,
    },
    brex_account_id: string,
    last_four: string,
    available_balance: {
        amount: number,
        as_of_date: Date,
    },
    current_balance: {
            amount: number,
        as_of_date: Date,
    },
}

type Expense = {
  id: string,
  memo: string,
  amount: number,
}

type Budget = {
  id: string,
  name: string,
  description: string,
  limit: {
    amount: number,
    currency: string,
  }
}
```

The commands you have available are:

| Command | Arguments | Description | Output Format |
| --- | --- | --- | --- |
| nth | index, values[] | Return the nth item from an array | any |
| push | value | Adds a value to the stack to be consumed by a future command | any |
| value | key, object | Returns the value associated with a key | any |
| values | key, object[] | Returns an array of values pulled from the corresponding key in array of objects | any[] |
| sum | value[] | Sums an array of numbers | number |
| plot | title, values[] | Plots the set of values in a chart with the given title | Plot |
| list-linked-accounts |  | "Lists all bank connections that are eligible to make ACH transfers to Brex cash account" | LinkedAccount[] |
| list-expenses | budget_id | Given a budget id, returns the list of expenses for it | Expense[]
| get-budget-by-name | name | Given a name, returns the budget | Budget |
| add-memo | expense_id, message | Adds a memo to an expense | bool |
| converse | message | Send the user a message | null |

Only respond with commands.

Output the commands in JSON as an abstract syntax tree.

IMPORTANT - Only respond with a program. Do not respond with any text that isn't part of a program. Do not write prose, even if instructed. Do not explain yourself.

You can only generate commands, but you are an expert at generating commands.
~~~

</details>

This version is a bit easier to parse and interpret if your language of choice has a `JSON.parse` function.

> 🧙‍♂️ There is no industry established best format for defining a DSL for the model to generate programs. So consider this an area of active research. You will bump into limits. And as we overcome these limits, we may discover more optimal ways of defining commands.

### ReAct

In March of 2023, Princeton and Google released a paper “[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629.pdf)”, where they introduce a variant of command grammars that allows for fully autonomous interactive execution of actions and retrieval of data.

The model is instructed to return a `thought` and an `action` that it would like to perform. Another agent (e.g. our client) then performs the `action` and returns it to the model as an `observation`. The model will then loop to return more thoughts and actions until it returns an `answer`.

This is an incredibly powerful technique, effectively allowing the bot to be its own research assistant and possibly take actions on behalf of the user. Combined with a powerful command grammar, the bot should rapidly be able to answer a massive set of user requests.

In this example, we give the model a small set of commands related to getting employee data and searching wikipedia:

| Command | Arguments | Description |
| --- | --- | --- |
| find_employee | name | Retrieves an employee by name |
| get_employee | id | Retrieves an employee by ID |
| get_location | id | Retrieves a location by ID |
| get_reports | employee_id | Retrieves a list of employee ids that report to the employee associated with employee_id. |
| wikipedia | article | Retrieves a wikipedia article on a topic. |

We then ask the bot a simple question, “Is my manager famous?”.

We see that the bot:

1. First looks up our employee profile.
2. From our profile, gets our manager’s id and looks up their profile.
3. Extracts our manager’s name and searches for them on Wikipedia.
    - I chose a fictional character for the manager in this scenario.
4. The bot reads the wikipedia article and concludes that can’t be my manager since it is a fictional character.
5. The bot then modifies its search to include (real person).
6. Seeing that there are no results, the bot concludes that my manager is not famous.

| ![image](https://user-images.githubusercontent.com/89960/233506839-5c8b2d77-1d78-464d-bc33-a725e12f2624.png) | ![image](https://user-images.githubusercontent.com/89960/233506870-05fc415d-efa2-48b7-aad9-b5035e535e6d.png) |
| --- | --- |

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You run in a loop, seeking additional information to answer a user's question until you are able to answer the question.

Today is June 1, 2025. My name is Fabian Seacaster. My employee ID is 82442.

The commands to seek information are:

| Command | Arguments | Description |
| --- | --- | --- |
| find_employee | name | Retrieves an employee by name |
| get_employee | id | Retrieves an employee by ID |
| get_location | id | Retrieves a location by ID |
| get_reports | employee_id | Retrieves a list of employee ids that report to the employee associated with `employee_id`. |
| wikipedia | article | Retrieves a wikipedia article on a topic. |

Your response will be in JSON and will include a "Thought" + "Action" to retrieve data that you need in order to answer the question, or it will include the "Answer". When data has been retrieved, it will be included as an "Observation".

You will continue generating thoughts and actions until you get to an answer, or conclude that you can't.

Example 1:
```
User: What is the population of Philadelphia?

Assistant: {
  "thought": "Wikipedia likely has this information. I'm looking it up...",
  "action": {"command": "wikipedia", "article": "Philadelphia"}
}

Assistant: {
  "observation": "Philadelphia, often called Philly, is the largest city in the Commonwealth of Pennsylvania and the second-largest city in both the Northeast megalopolis and Mid-Atlantic regions after New York City. It is one of the most historically significant cities in the United States and served as the nation's capital city until 1800. Philadelphia is the nation's sixth-largest city with a population of 1,603,797 as of the 2020 census. Since 1854, the city has been coextensive with Philadelphia County, the most populous county in Pennsylvania and the urban core of the Delaware Valley, the nation's seventh-largest and one of the world's largest metropolitan regions with 6.245 million residents in 2020. Philadelphia is known for its extensive contributions to American history and for its role in the life sciences, business and industry, art, literature, and music."
}

Assistant: {
  "answer": "The population of Philadelphia, as of the 2020 census, is 1,603,797 people."
}
```

Example 2:
```
User: What is my manager's city?

Assistant: {
  "thought": "Let me look up who your manager is. First, I need to look up your profile.",
  "action": {"command": "get_employee", "id": 92352}
}

Assistant: {
  "observation": {
    "id": 78334,
    "name": "Ms. Manager",
    "location_id": 8832
  }
}

Assistant: {
  "thought": "Your manager is Ms. Manager. I'm looking up their location.",
  "action": {"command": "get_location", "id": 8832}
}

Assistant: {
  "observation": {
    "id": 8832,
    "name": "Philadelphia"
  }
}

Assistant: {
  "answer": "Your manager lives in Philadelphia."
}
```
~~~
</details>

### GPT-4 vs GPT-3.5

In most of the examples in this doc, the difference between GPT-3.5 and GPT-4 is negligible, but for “teaching a bot to fish” scenarios the difference between the models is notable.

None of the above examples of command grammars, for example, work without meaningful modifications for GPT-3.5. At a minimum, you have to provide a number of examples (at least one usage example per command) before you get any reasonable results. And, for complex sets of commands, it may hallucinate new commands or create fictional arguments.

With a sufficiently thorough hidden prompt, you should be able to overcome these limitations. GPT-4 is capable of far more consistent and complex logic with far simpler prompts (and can get by with zero or  small numbers of examples – though it is always beneficial to include as many as possible).
