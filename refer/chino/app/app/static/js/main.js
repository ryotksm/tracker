$(function() {
  function train() {
    $('#button-train').prop('disabled', true)

    $.ajax({url: '/train'})
    .then(
      (data) => {
        status_loop();
        set_chino_state(2)
      },
      () => {
        $('#button-train').prop('disabled', false)
      }
    );
  }

  $('#button-train').click(function(){
    train()
  });

  function set_chino_state(state) {
    for (var i = 1; i <= 5; i++) {
      s = 'state-' + i
      if ($('#chino-image').hasClass(s)) {
        $('#chino-image').removeClass(s)
      }
    }

    $('#chino-image').addClass('state-' + state)

    if (state == 3) {
      $.ajax({
        url: '/labels',
        type : "GET"})
        .then(
          (labels) => {
            label_names = labels.map(function(x) { return "「" + x + "」";});
            $('#message').html("学習が終わったよ。<br/> 「判定する画像を選択」ボタンから" + label_names.join('か') + "の画像を選んでね！");
          },
          () => {
          }
        );
    }
  }


  function preview_image(input) {
    var reader = new FileReader();

    reader.onload = function (e) {
        $('#preview-image').css('background', 'no-repeat url(' + e.target.result + ')');
        $('#preview-image').css('background-size', 'cover');
    }
    reader.readAsDataURL(input.files[0]);
  }

  $('#preview-image').css('background-size', 'cover');
  $("#select-image").change(function(){
    preview_image(this);
    infer_upload();
  });

  function infer_upload() {
    var formdata = new FormData($('#infer-upload').get(0));
    $('#message').text('(判定中…)');

    $.ajax({
      url: '/infer',
      type : "POST",
      data: formdata,
      contentType : false,
      processData : false,
      dataType: "html"})
      .then(
        (status) => {
          set_chino_state(4)
          $('#message').text('これは' + status + 'だね');

          setTimeout(function() {
            update_status()
          }, 7000);
        },
        (err) => {
          if (err.status == 500) {
            $('#message').text('画像ファイル(jpg, png, gif)を選択してね！');
            set_chino_state(5)

            setTimeout(function() {
              update_status()
            }, 7000);
          } else if (err.status == 400) {
            $('#message').text('まだ学習してないよ！');
            set_chino_state(5)

            setTimeout(function() {
                update_status()
            }, 7000);
          }
        }
      );
  }

  function preview_dataset() {
    $.ajax({url: '/find_dataset'})
    .then(
      (data) => {
        $('#dataset-preview').html('')

        for (var label in data) {
          paths = data[label]
          column = $('<div/>', {class: 'column is-full'})
          subcolumns = $('<div/>', {class: 'columns is-multiline is-mobile'})

          label_p = $('<h2/>', {class: 'title is-6'})
          label_p.text(label + " (" + paths.length + "枚)")
          column.append(label_p)

          for (var i = 0; i < paths.length && i <= 2; i++) {
            var path = paths[i]
            subcolumn = $('<div/>', {class: 'column is-one-third'})
            fig = $('<figure/>', {class: 'image is-1by1'})
            img = $('<img/>', {alt: path, src: path})
            fig.append(img)
            subcolumn.append(fig)
            subcolumns.append(subcolumn)
          }
          column.append(subcolumns)

          $('#dataset-preview').append(column)
        }
      },
      () => {
        $('#button-train').prop('disabled', false)
      }
    );
  }
  $("#button-find-dataset").change(function(){
    preview_dataset();
  });

  function loop() {
    preview_dataset();

    setTimeout(function() {
      loop()
    }, 1000);
  }

  function chino_loop() {
    $('#chino-image').toggleClass('alt')

    setTimeout(function() {
      chino_loop()
    }, 1000);
  }

  status_pool_interval = null

  function status_loop() {
    poll_status()

    status_pool_interval = setTimeout(function() {
      status_loop()
    }, 500);
  }

  function poll_status() {
    $.ajax({url: '/status'})
      .then(
        (status) => {
          handle_status(status);
        },
        (error) => {
        }
      );
  }

  function handle_status(status) {
    if(status['training_phase'] == 10) {
      perc = Math.floor(status['progress']*100);
      $('#message').text("学習中…\n" + '(' + perc + '% 完了)')
    } else if(status['training_phase'] == 20) {
      clearInterval(status_pool_interval);
      set_chino_state(3)
      $('#button-train').prop('disabled', false)
    } else {
      clearInterval(status_pool_interval);
      set_chino_state(5)
      $('#button-train').prop('disabled', false)
    }
  }

  function update_status() {
    $.ajax({url: '/status'})
      .then(
        (status) => {
          console.log(status)
          if (status['training_phase'] == 10) {
            status_loop();
            set_chino_state(2)
          } else if(status['training_phase'] == 20) {
            $('#button-train').prop('disabled', false)
            set_chino_state(3)
          } else if(status['training_phase'] == 1) {
            $('#message').text('「学習をスタート」ボタンを押してね！')
            $('#button-train').prop('disabled', false)
            set_chino_state(1)
          }
        },
        (error) => {
          console.log(error)
        }
      );
  }

  update_status();

  $('#button-train').prop('disabled', true)

  loop()
  chino_loop();
});
