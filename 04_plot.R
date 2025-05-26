#!/usr/bin/env Rscript

# plot_annotations_manual.R
# Reads all *_annotated.csv files and plots the % of runs per model
# using the *_annotated_manual justification columns.

library(tidyverse)
library(grid)    # for unit()

# 1. Gather & read only the manual‐annotation columns
files <- c(
  "analogy_responses_anthropic_annotated.csv",
  "analogy_responses_openai_annotated.csv",
  "analogy_responses_ollama_annotated.csv"
) %>% keep(file.exists)

if (!length(files)) stop("No annotated CSVs found.")

# Note: .id goes on map_dfr(), not read_csv()
df <- map_dfr(
  set_names(files),
  ~ read_csv(.x,
             col_types = cols_only(
               model                                   = col_character(),
               run                                     = col_integer(),
               coordination_justification_annotated_manual   = col_integer(),
               optimal_move_justification_annotated_manual   = col_integer(),
               convergence_justification_annotated_manual    = col_integer()
             )
  ),
  .id = "source"
) %>%
  # 2. Rename the long column names to simple flags
  rename(
    coordination = coordination_justification_annotated_manual,
    optimal_move = optimal_move_justification_annotated_manual,
    convergence  = convergence_justification_annotated_manual
  )

# 3. Reshape & summarize
summary_df <- df %>%
  pivot_longer(coordination:convergence,
               names_to  = "annotation",
               values_to = "flag") %>%
  group_by(model, annotation) %>%
  summarize(pct_true = mean(flag) * 100, .groups = "drop")

# 4. Plot polished, horizontal bars
p <- ggplot(summary_df, aes(x = model, y = pct_true, fill = annotation)) +
  geom_col(color = "white", size = 0.2,
           position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = sprintf("%.0f%%", pct_true)),
            position = position_dodge(width = 0.7),
            hjust = -0.1, size = 3) +
  scale_fill_brewer(
    palette = "Set2", name = NULL,
    breaks = c("coordination", "optimal_move", "convergence"),
    labels = c("Coordination", "Optimal move", "Convergence")
  ) +
  scale_y_continuous(
    labels = scales::percent_format(scale = 1),
    expand = expansion(mult = c(0, 0.15))
  ) +
  coord_flip() +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "grey90", size = 0.4),
    legend.position    = "bottom",
    legend.direction   = "horizontal",
    legend.key.width   = unit(1.2, "lines"),
    axis.text.y        = element_text(face = "bold", size = 12),
    axis.text.x        = element_text(size = 10),
    axis.ticks         = element_blank(),
    plot.margin        = margin(5, 20, 5, 5)
  )

# 5. Save & display
ggsave("annotation_summary_all_models.png", p,
       width = 8, height = 5, dpi = 300, bg = "white")
print(p)
