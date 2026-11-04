def simulate_cards():
    cards_html = ['<div class="worker-grid">']
    for i in range(2):
        card = f"""
            <div class="worker-card">
              <div class="worker-header">
                <span class="worker-name">Worker #{i:02d}</span>
                <span class="status-pill pill-slate">IDLE</span>
              </div>
              <div class="worker-task" title="Idle">Idle</div>
              <div class="worker-stats">
                <span>Độ trễ: <b>0ms</b></span>
                <span>Thành công: <b>0</b> / 0</span>
              </div>
            </div>
            """
        cards_html.append(card)
    cards_html.append("</div>")
    joined = "".join(cards_html)
    print("Joined output:")
    print(joined)

if __name__ == "__main__":
    simulate_cards()
