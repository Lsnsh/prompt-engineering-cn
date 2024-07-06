# Strategies

This section contains examples and strategies for specific needs or problems. For successful prompt engineering, you will need to combine some subset of all of the strategies enumerated in this document. Don’t be afraid to mix and match things – or invent your own approaches.

## Embedding Data

In hidden contexts, you’ll frequently want to embed all sorts of data. The specific strategy will vary depending on the type and quantity of data you are embedding.

### Simple Lists

For one-off objects, enumerating fields + values in a normal bulleted list works pretty well:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507156-0bdbc0af-d977-44e0-a8d5-b30538c5bbd9.png" title="GPT-4 extracting Steve’s occupation from a list attributes.">
</p>

It will also work for larger sets of things, but there are other formats for lists of data that GPT handles more reliably. Regardless, here’s an example:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507223-9cda591e-62f3-4339-b227-a07c37b90724.png" title="GPT-4 answering questions about a set of expenses.">
</p>

### Markdown Tables

Markdown tables are great for scenarios where you have many items of the same type to enumerate.

Fortunately, OpenAI’s models are exceptionally good at working with Markdown tables (presumably from the tons of GitHub data they’ve trained on).

We can reframe the above using Markdown tables instead:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507313-7ccd825c-71b9-46d3-80c9-30bf97a8e090.png" title="GPT-4 answering questions about a set of expenses from a Markdown table.">
</p>

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507395-b8ecb641-726c-4e57-b85e-13f6b7717f22.png" title="GPT-4 answering questions about a set of expenses from a Markdown table.">
</p>

> 🧠 Note that in this last example, the items in the table have an explicit date, February 2nd. In our question, we asked about “today”. And earlier in the prompt we mentioned that today was Feb 2. The model correctly handled the transitive inference – converting “today” to “February 2nd” and then looking up “February 2nd” in the table.

### JSON

Markdown tables work really well for many use cases and should be preferred due to their density and ability for the model to handle them reliably, but you may run into scenarios where you have many columns and the model struggles with it or every item has some custom attributes and it doesn’t make sense to have dozens of columns of empty data.

In these scenarios, JSON is another format that the model handles really well. The close proximity of `keys` to their `values` makes it easy for the model to keep the mapping straight.

Here is the same example from the Markdown table, but with JSON instead:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507559-26e6615d-4896-4a2c-b6ff-44cbd7d349dc.png" title="GPT-4 answering questions about a set of expenses from a JSON blob.">
</p>

### Freeform Text

Occasionally you’ll want to include freeform text in a prompt that you would like to delineate from the rest of the prompt – such as embedding a document for the bot to reference. In these scenarios, surrounding the document with triple backticks, ```, works well[^8].

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507684-93222728-e216-47b4-8554-04acf9ec6201.png" title="GPT-4 answering questions about a set of expenses from a JSON blob.">
</p>

[^8]: A good rule of thumb for anything you’re doing in prompts is to lean heavily on things the model would have learned from GitHub.

### Nested Data

Not all data is flat and linear. Sometimes you’ll need to embed data that is nested or has relations to other data. In these scenarios, lean on `JSON`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507758-7baffcaa-647b-4869-9cfb-a7cf8849c453.png" title="GPT-4 handles nested JSON very reliably.">
</p>

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You answer questions about users. Here is what you know about them:

{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "contact": {
        "address": {
          "street": "123 Main St",
          "city": "Anytown",
          "state": "CA",
          "zip": "12345"
        },
        "phone": "555-555-1234",
        "email": "johndoe@example.com"
      }
    },
    {
      "id": 2,
      "name": "Jane Smith",
      "contact": {
        "address": {
          "street": "456 Elm St",
          "city": "Sometown",
          "state": "TX",
          "zip": "54321"
        },
        "phone": "555-555-5678",
        "email": "janesmith@example.com"
      }
    },
    {
      "id": 3,
      "name": "Alice Johnson",
      "contact": {
        "address": {
          "street": "789 Oak St",
          "city": "Othertown",
          "state": "NY",
          "zip": "67890"
        },
        "phone": "555-555-2468",
        "email": "alicejohnson@example.com"
      }
    },
    {
      "id": 4,
      "name": "Bob Williams",
      "contact": {
        "address": {
          "street": "135 Maple St",
          "city": "Thistown",
          "state": "FL",
          "zip": "98765"
        },
        "phone": "555-555-8642",
        "email": "bobwilliams@example.com"
      }
    },
    {
      "id": 5,
      "name": "Charlie Brown",
      "contact": {
        "address": {
          "street": "246 Pine St",
          "city": "Thatstown",
          "state": "WA",
          "zip": "86420"
        },
        "phone": "555-555-7531",
        "email": "charliebrown@example.com"
      }
    },
    {
      "id": 6,
      "name": "Diane Davis",
      "contact": {
        "address": {
          "street": "369 Willow St",
          "city": "Sumtown",
          "state": "CO",
          "zip": "15980"
        },
        "phone": "555-555-9512",
        "email": "dianedavis@example.com"
      }
    },
    {
      "id": 7,
      "name": "Edward Martinez",
      "contact": {
        "address": {
          "street": "482 Aspen St",
          "city": "Newtown",
          "state": "MI",
          "zip": "35742"
        },
        "phone": "555-555-6813",
        "email": "edwardmartinez@example.com"
      }
    },
    {
      "id": 8,
      "name": "Fiona Taylor",
      "contact": {
        "address": {
          "street": "531 Birch St",
          "city": "Oldtown",
          "state": "OH",
          "zip": "85249"
        },
        "phone": "555-555-4268",
        "email": "fionataylor@example.com"
      }
    },
    {
      "id": 9,
      "name": "George Thompson",
      "contact": {
        "address": {
          "street": "678 Cedar St",
          "city": "Nexttown",
          "state": "GA",
          "zip": "74125"
        },
        "phone": "555-555-3142",
        "email": "georgethompson@example.com"
      }
    },
    {
      "id": 10,
      "name": "Helen White",
      "contact": {
        "address": {
          "street": "852 Spruce St",
          "city": "Lasttown",
          "state": "VA",
          "zip": "96321"
        },
        "phone": "555-555-7890",
        "email": "helenwhite@example.com"
      }
    }
  ]
}
~~~
</details>

If using nested `JSON` winds up being too verbose for your token budget, fallback to `relational tables` defined with `Markdown`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507968-a378587b-e468-4882-a1e8-678d9f3933d3.png" title="GPT-4 handles relational tables pretty reliably too.">
</p>

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You answer questions about users. Here is what you know about them:

Table 1: users
| id (PK) | name          |
|---------|---------------|
| 1       | John Doe      |
| 2       | Jane Smith    |
| 3       | Alice Johnson |
| 4       | Bob Williams  |
| 5       | Charlie Brown |
| 6       | Diane Davis   |
| 7       | Edward Martinez |
| 8       | Fiona Taylor  |
| 9       | George Thompson |
| 10      | Helen White   |

Table 2: addresses
| id (PK) | user_id (FK) | street      | city       | state | zip   |
|---------|--------------|-------------|------------|-------|-------|
| 1       | 1            | 123 Main St | Anytown    | CA    | 12345 |
| 2       | 2            | 456 Elm St  | Sometown   | TX    | 54321 |
| 3       | 3            | 789 Oak St  | Othertown  | NY    | 67890 |
| 4       | 4            | 135 Maple St | Thistown  | FL    | 98765 |
| 5       | 5            | 246 Pine St | Thatstown  | WA    | 86420 |
| 6       | 6            | 369 Willow St | Sumtown  | CO    | 15980 |
| 7       | 7            | 482 Aspen St | Newtown   | MI    | 35742 |
| 8       | 8            | 531 Birch St | Oldtown   | OH    | 85249 |
| 9       | 9            | 678 Cedar St | Nexttown  | GA    | 74125 |
| 10      | 10           | 852 Spruce St | Lasttown | VA    | 96321 |

Table 3: phone_numbers
| id (PK) | user_id (FK) | phone       |
|---------|--------------|-------------|
| 1       | 1            | 555-555-1234 |
| 2       | 2            | 555-555-5678 |
| 3       | 3            | 555-555-2468 |
| 4       | 4            | 555-555-8642 |
| 5       | 5            | 555-555-7531 |
| 6       | 6            | 555-555-9512 |
| 7       | 7            | 555-555-6813 |
| 8       | 8            | 555-555-4268 |
| 9       | 9            | 555-555-3142 |
| 10      | 10           | 555-555-7890 |

Table 4: emails
| id (PK) | user_id (FK) | email                 |
|---------|--------------|-----------------------|
| 1       | 1            | johndoe@example.com   |
| 2       | 2            | janesmith@example.com |
| 3       | 3            | alicejohnson@example.com |
| 4       | 4            | bobwilliams@example.com |
| 5       | 5            | charliebrown@example.com |
| 6       | 6            | dianedavis@example.com |
| 7       | 7            | edwardmartinez@example.com |
| 8       | 8            | fionataylor@example.com |
| 9       | 9            | georgethompson@example.com |
| 10      | 10           | helenwhite@example.com |

Table 5: cities
| id (PK) | name         | state | population | median_income |
|---------|--------------|-------|------------|---------------|
| 1       | Anytown     | CA    | 50,000     | $70,000      |
| 2       | Sometown    | TX    | 100,000    | $60,000      |
| 3       | Othertown   | NY    | 25,000     | $80,000      |
| 4       | Thistown    | FL    | 75,000     | $65,000      |
| 5       | Thatstown   | WA    | 40,000     | $75,000      |
| 6       | Sumtown     | CO    | 20,000     | $85,000      |
| 7       | Newtown     | MI    | 60,000     | $55,000      |
| 8       | Oldtown     | OH    | 30,000     | $70,000      |
| 9       | Nexttown    | GA    | 15,000     | $90,000      |
| 10      | Lasttown    | VA    | 10,000     | $100,000     |
~~~

</details>

> 🧠 The model works well with data in [3rd normal form](https://en.wikipedia.org/wiki/Third_normal_form), but may struggle with too many joins. In experiments, it seems to do okay with at least three levels of nested joins. In the example above the model successfully joins from `users` to `addresses` to `cities` to infer the likely income for George – $90,000.

## Citations

Frequently, a natural language response isn’t sufficient on its own and you’ll want the model’s output to cite where it is getting data from. 

One useful thing to note here is that anything you might want to cite should have a unique ID. The simplest approach is to just ask the model to link to anything it references:


<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509069-1dcbffa2-8357-49b5-be43-9791f93bd0f8.png" title="GPT-4 will reliably link to data if you ask it to.">
</p>

## Programmatic Consumption

By default, language models output natural language text, but frequently we need to interact with this result in a programmatic way that goes beyond simply printing it out on screen. You can achieve this by  asking the model to output the results in your favorite serialization format (JSON and YAML seem to work best).

Make sure you give the model an example of the output format you’d like. Building on our previous travel example above, we can augment our prompt to tell it:

~~~
Produce your output as JSON. The format should be:
```
{
    message: "The message to show the user",
    hotelId: 432,
    flightId: 831
}
```

Do not include the IDs in your message.
~~~

And now we’ll get interactions like this:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509174-be0c3bc5-08e3-4d1a-8841-52c401def770.png" title="GPT-4 providing travel recommendations in an easy to work with format.">
</p>

You could imagine the UI for this rendering the message as normal text, but then also adding discrete buttons for booking the flight + hotel, or auto-filling a form for the user.

As another example, let’s build on the [citations](#citations) example – but move beyond Markdown links. We can ask it to produce JSON with a normal message along with a list of items used in the creation of that message. In this scenario you won’t know exactly where in the message the citations were leveraged, but you’ll know that they were used somewhere.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509280-59d9ff46-0e95-488a-b314-a7d2b7c9bfa3.png" title="Asking the model to provide a list of citations is a reliable way to programmatically know what data the model leaned on in its response.">
</p>

> 🧠 Interestingly, in the model’s response to “How much did I spend at Target?” it provides a single value, $188.16, but **importantly** in the `citations` array it lists the individual expenses that it used to compute that value.

## Chain of Thought

Sometimes you will bang your head on a prompt trying to get the model to output reliable results, but, no matter what you do, it just won’t work. This will frequently happen when the bot’s final output requires intermediate thinking, but you ask the bot only for the output and nothing else.

The answer may surprise you: ask the bot to show its work. In October 2022, Google released a paper “[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)” where they showed that if, in your hidden prompt, you give the bot examples of answering questions by showing your work, then when you ask the bot to answer something it will show its work and produce more reliable answers.

Just a few weeks after that paper was published, at the end of October 2022, the University of Tokyo and Google released the paper “[Large Language Models are Zero-Shot Reasoners](https://openreview.net/pdf?id=e2TBb5y0yFf)”, where they show that you don’t even need to provide examples – **you simply have to ask the bot to think step-by-step**.

### Averaging

Here is an example where we ask the bot to compute the average expense, excluding Target. The actual answer is $136.77 and the bot almost gets it correct with $136.43.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509534-2b32c8dd-a1ee-42ea-82fb-4f84cfe7e9ba.png" title="The model **almost** gets the average correct, but is a few cents off.">
</p>

If we simply add “Let’s think step-by-step”, the model gets the correct answer:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509608-6e53995b-668b-47f6-9b5e-67afad89f8bc.png" title="When we ask the model to show its work, it gets the correct answer.">
</p>

### Interpreting Code

Let’s revisit the Python example from earlier and apply chain-of-thought prompting to our question. As a reminder, when we asked the bot to evaluate the Python code it gets it slightly wrong. The correct answer is `Hello, Brex!!Brex!!Brex!!!` but the bot gets confused about the number of !'s to include. In below’s example, it outputs `Hello, Brex!!!Brex!!!Brex!!!`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509724-8f3302f8-59eb-4d3b-8939-53d7f63b0299.png" title="The bot almost interprets the Python code correctly, but is a little off.">
</p>

If we ask the bot to show its work, then it gets the correct answer:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509790-2a0f2189-d864-4d27-aacb-cfc936fad907.png" title="The bot correctly interprets the Python code if you ask it to show its work.">
</p>

### Delimiters

In many scenarios, you may not want to show the end user all of the bot’s thinking and instead just want to show the final answer. You can ask the bot to delineate the final answer from its thinking. There are many ways to do this, but let’s use JSON to make it easy to parse:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509865-4f3e7265-6645-4d43-8644-ecac5c0ca4a7.png" title="The bot showing its work while also delimiting the final answer for easy extraction.">
</p>

Using Chain-of-Thought prompting will consume more tokens, resulting in increased price and latency, but the results are noticeably more reliable for many scenarios. It’s a valuable tool to use when you need the bot to do something complex and as reliably as possible.

## Fine Tuning

Sometimes no matter what tricks you throw at the model, it just won’t do what you want it to do. In these scenarios you can **sometimes** fallback to fine-tuning. This should, in general, be a last resort.

[Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) is the process of taking an already trained model and then giving it thousands (or more) of example `input:output` pairs

It does not eliminate the need for hidden prompts, because you still need to embed dynamic data, but it may make the prompts smaller and more reliable.

### Downsides

There are many downsides to fine-tuning. If it is at all possible, take advantage of the nature of language models being [zero-shot, one-shot, and few-shot learners](https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing)) by teaching them to do something in their prompt rather than fine-tuning.

Some of the downsides include:

- **Not possible**: [GPT-3.5/GPT-4 isn’t fine tunable](https://platform.openai.com/docs/guides/chat/is-fine-tuning-available-for-gpt-3-5-turbo), which is the primary model / API we’ll be using, so we simply can’t lean in fine-tuning.
- **Overhead**: Fine-tuning requires manually creating tons of data.
- **Velocity**: The iteration loop becomes much slower – every time you want to add a new capability, instead of adding a few lines to a prompt, you need to create a bunch of fake data and then run the finetune process and then use the newly fine-tuned model.
- **Cost**: It is up to 60x more expensive to use a fine-tuned GPT-3 model vs the stock `gpt-3.5-turbo` model. And it is 2x more expensive to use a fine-tuned GPT-3 model vs the stock GPT-4 model.

> ⛔️ If you fine-tune a model, **never use real customer data**. Always use synthetic data. The model may memorize portions of the data you provide and may regurgitate private data to other users that shouldn’t be seeing it.
>
> If you never fine-tune a model, we don’t have to worry about accidentally leaking data into the model.
