from matplotlib import pyplot as plt

with open("./BMCModel/rnn/result/life_losses_result_ST12000NM0008_RNN-20.txt", "r", encoding='utf-8') as f:
    v_losses = eval(f.readline().strip())
    t_losses = eval(f.readline().strip())
    best_model_no = v_losses.index(min(v_losses))

plt.plot(t_losses, label="Training loss")
plt.plot(v_losses, label="Validation loss")
# plt.ylim(min(min(t_losses),min(v_losses)), max(max(t_losses),max(v_losses)))
plt.ylim(0,0.2)
plt.xlim(-1, 50)
plt.vlines(best_model_no, 0, max(v_losses[best_model_no], 0.1), colors='gray', linestyles='dashed',
           label="Best:" + str(best_model_no))
plt.legend()
plt.title("Losses-RNN")
plt.savefig('./result/RNN.png')
plt.show()
