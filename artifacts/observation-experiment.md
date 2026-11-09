# Observation experiment

Comparison of the four saved trajectories and result files. White is the model; Black is the deterministic server bot. Each condition has only one run.

| Model | Legal moves shown? | Recorded outcome | Completed full moves | Model responses | Total `play_move` calls | Rejected illegal moves | Invalid-move rate | Transport errors | `game_over` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `deepseek-v4-flash` | No | Black wins by checkmate | 27 | 33 | 33 | 4 | 12.12% | 2 | true |
| `deepseek-v4-flash` | Yes | Black wins by checkmate | 27 | 28 | 28 | 1 | 3.57% | 0 | true |
| `gpt-oss-120b` | No | Unfinished; White to move | 21 | 26 | 21 | 0 | 0% | 0 | false |
| `gpt-oss-120b` | Yes | Black wins by checkmate | 13 | 13 | 13 | 0 | 0% | 0 | true |

Invalid-move rate is rejected illegal `play_move` calls divided by all `play_move` calls, including calls that encountered transport errors. Transport errors are reported separately and are not counted as illegal moves. Calls are counted once from `responses`, rather than from repeated prompt snapshots. The `game_over` values come from the corresponding result files.

## Comparison

- **DeepSeek:** Showing legal moves reduced illegal requests from four to one, but both games ended in defeat after 27 moves. The no-legal run also encountered two recoverable transport failures, so not all extra responses reflect chess reasoning errors.
- **GPT-OSS:** Neither condition produced server-rejected moves. However, without legal moves, four responses hit the output-length limit without a tool call. The final response emitted `<play_move>...` as text rather than an actual function call. Its saved game remains active after 21 moves; this is not a win or draw. With legal moves shown, every response called the tool, but White was checkmated after 13 moves.
- **Reasoning cost:** Recorded completion tokens fell from 9,788 to 7,812 for DeepSeek and from 65,952 to 18,978 for GPT-OSS when legal moves were shown. These are total-run counts, not per-move comparisons; the GPT-OSS games also had different lengths.

Overall, legal-move observations improved request reliability in these runs, but did not ensure good chess decisions or a win. The unfinished GPT-OSS run cannot be treated as stronger play merely because it lasted longer. More repeated games would be needed for a reliable performance conclusion.

Sources: `part3-{no-legal-moves,legal-moves}-{deepseek,gpt-oss}.json` and their corresponding `-result.json` files in this directory. Error counts use tool observations in each trajectory's final prompt snapshot to avoid counting repeated conversation history.
