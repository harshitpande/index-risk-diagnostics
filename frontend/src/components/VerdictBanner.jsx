import SignalStatusBar from "./SignalStatusBar";
import "./VerdictBanner.css";

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  year: "numeric",
  month: "short",
  day: "numeric",
});

// Current-state headline: today's rule-based regime, why it was assigned, how long it has held,
// and the three early-warning signal lights. All values are direct reads of snapshot.json.
export default function VerdictBanner({ snapshot, error }) {
  if (error) {
    return (
      <section className="verdict-banner verdict-banner--empty">
        <span>Couldn&rsquo;t load current risk state: {error}</span>
      </section>
    );
  }

  if (!snapshot) {
    return (
      <section className="verdict-banner verdict-banner--empty">
        <span>Loading current risk state&hellip;</span>
      </section>
    );
  }

  const regimeKey = snapshot.current_regime.toLowerCase();

  return (
    <section
      className="verdict-banner"
      style={{
        // Text variant for the accent too: the Crisis line colour is too dark to read on the card.
        "--verdict-accent": `var(--color-regime-${regimeKey}-text)`,
        "--verdict-text": `var(--color-regime-${regimeKey}-text)`,
      }}
      aria-label="Current risk state"
    >
      <div className="verdict-banner-top">
        <h2 className="verdict-banner-regime">{snapshot.current_regime}</h2>
        <SignalStatusBar signals={snapshot.signals} />
      </div>
      <p className="verdict-banner-meta">
        {snapshot.days_in_regime} trading days in this regime, as of{" "}
        {dateFormatter.format(new Date(snapshot.date))}
      </p>
      <p className="verdict-banner-reasoning">{snapshot.reasoning}</p>
    </section>
  );
}
