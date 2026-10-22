import { Link } from 'react-router-dom';
import { type Method as MethodDoc, api, useGet } from '../lib/api';
import { DASH, int, num, sci } from '../lib/format';
import { LineChart, Loading, NotYet, PurposeBadge, TokensView } from '../lib/ui';
import { dataVersionPath, rowPath, trainRunPath } from '../lib/url';

type Cfg = Record<string, unknown>;
const pick = (cfg: Cfg | undefined, section: string, key: string): unknown => {
  const s = cfg?.[section];
  return s && typeof s === 'object' ? (s as Cfg)[key] : undefined;
};
const show = (v: unknown) => (v == null ? DASH : typeof v === 'number' ? (v < 0.01 && v > 0 ? sci(v) : String(v)) : String(v));

/** `/method`: how it was trained, on the latest real job: the recipe, its loss, and one row as trained. */
export function Method() {
  const { data, error } = useGet<MethodDoc>(api.method());
  if (!data) return <Loading error={error} what="the method" />;
  const run = data.train_run;
  const cfg = run?.config as Cfg | undefined;
  const ds = data.dataset;
  return (
    <>
      <p className="eyebrow">Method</p>
      <h1>
        One conversation is one row, and every agent turn in it is a target
      </h1>
      <p className="lede">
        Supervised fine-tuning copies a teacher. The rows are the board's clean passing conversations as the agent saw them: tau2's stock
        system prompt and 13 tools, the customer's words, the agent's turns and its own tool results. The loss is on the agent's turns only, so
        the student learns to write what the teacher wrote, given exactly the history the teacher had.
      </p>
      <ol className="steps">
        <li className="step">
          <b>Read the whole row once.</b> The chat template renders it into one sequence
          {ds ? ` (median ${int(ds.tokens.median)} tokens in ${ds.version}; ${int(ds.tokens.prompt_and_tools)} of them are the prompt and tools)` : ''}.
        </li>
        <li className="step">
          <b>Guess the next token at every position.</b> One forward pass gives a probability for each of the vocabulary's tokens at every
          position. Attention looks backwards only, so the guess at an agent token uses only what came before it.
        </li>
        <li className="step">
          <b>Score only the agent's tokens.</b> Every other position's label is −100: the system prompt, the customer, the tool results. TRL's
          training copy of the chat template marks each agent turn with generation tags, and <code>assistant_only_loss</code> builds the mask
          from them.
        </li>
        <li className="step">
          <b>Update only the adapter.</b> LoRA adds small low-rank matrices to every linear layer; the base model stays frozen.
        </li>
      </ol>

      <section aria-labelledby="recipe-h">
        <span className="eyebrow">The recipe, read from the latest job's config</span>
        <h2 id="recipe-h">
          {run ? (
            <>
              <Link to={trainRunPath(run.run_id)}>{String(run.label ?? run.run_id)}</Link>{' '}
              <PurposeBadge purpose={(run.purpose as 'smoke' | 'result') ?? 'smoke'} />
            </>
          ) : (
            'No job yet'
          )}
        </h2>
        {!run ? (
          <NotYet what="training run" by="M2's make train-smoke" />
        ) : (
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>setting</th>
                  <th>value</th>
                  <th>what it does</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['base model', run.base_model ?? cfg?.base_model, 'the student whose weights stay frozen'],
                    ['tokenizer and template', run.tokenizer ?? cfg?.tokenizer, "Qwen3-4B-Instruct-2507's: the real student's chat format"],
                    ['dataset', run.dataset, 'the rows, by version and sha256'],
                    ['max_length', pick(cfg, 'sft', 'max_length'), 'the longest row trained, in tokens'],
                    ['assistant_only_loss', pick(cfg, 'sft', 'assistant_only_loss'), "the loss is on the agent's turns only"],
                    ['learning_rate', pick(cfg, 'sft', 'learning_rate'), 'the size of each update'],
                    ['lr_scheduler_type', pick(cfg, 'sft', 'lr_scheduler_type'), 'how the rate decays'],
                    ['num_train_epochs', pick(cfg, 'sft', 'num_train_epochs'), 'passes over the rows'],
                    ['gradient_accumulation_steps', pick(cfg, 'sft', 'gradient_accumulation_steps'), 'rows per update (batch 1)'],
                    ['LoRA r / alpha', `${show(pick(cfg, 'lora', 'r'))} / ${show(pick(cfg, 'lora', 'lora_alpha'))}`, "the adapter's rank and scale"],
                    ['target_modules', pick(cfg, 'lora', 'target_modules'), 'which layers get an adapter'],
                  ] as [string, unknown, string][]
                ).map(([k, v, why]) => (
                  <tr key={k}>
                    <td className="mono">{k}</td>
                    <td className="mono">{k === 'dataset' && v ? <Link to={dataVersionPath(String(v))}>{String(v)}</Link> : show(v)}</td>
                    <td>{why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="loss-h">
        <span className="eyebrow">The real loss curve</span>
        <h2 id="loss-h">Loss per optimiser step</h2>
        {data.metrics.length === 0 ? (
          <NotYet what="loss curve" by="a training job" />
        ) : (
          <LineChart
            title="Training loss"
            series={[{ name: 'loss', points: data.metrics.filter((m) => m.loss != null).map((m) => [m.step, m.loss as number]) }]}
            caption={`Final loss ${num(data.metrics.filter((m) => m.loss != null).at(-1)?.loss)} after ${data.metrics.at(-1)?.step ?? 0} steps.`}
          />
        )}
      </section>

      <section aria-labelledby="row-h">
        <span className="eyebrow">One real row, a target at every agent token</span>
        <h2 id="row-h">The exact text the job trained on, with the trained tokens marked</h2>
        {!data.mask ? (
          <NotYet what="masked row" by="a training job's mask check" />
        ) : (
          <>
            <p className="sub">
              Row{' '}
              <Link to={rowPath(String(run?.dataset ?? 'v1'), data.mask.row_sim_id, 'tokens')}>{data.mask.row_sim_id.slice(0, 8)}</Link>:{' '}
              {data.mask.check.trained_turns ?? DASH} agent turns, {int(data.mask.n_trained)} of {int(data.mask.n_tokens)} tokens trained.
            </p>
            <TokensView segments={data.mask.segments} nTokens={data.mask.n_tokens} nTrained={data.mask.n_trained} />
          </>
        )}
      </section>
    </>
  );
}
