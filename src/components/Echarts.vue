<template>
  <el-row :gutter="12">
    <el-col :span="18">
      <div id="main" style="min-height: 300px; width: 100%"></div
    ></el-col>
    <el-col :span="4">
      <div id="pie" style="min-height: 300px; width: 100%"></div>
    </el-col>
  </el-row>
</template>

<script>
import * as echarts from "echarts";

export default {
  props: {
    files: {
      required: true,
      type: Array,
    },
    labels: {
      required: true,
      type: Array,
    },
    predectedList: {
      required: true,
      type: Array,
    },
  },
  data() {
    return {
      main: null,
      pie: null,
      data: [],
    };
  },
  mounted() {
    this.main = echarts.init(document.getElementById("main"));
    this.pie = echarts.init(document.getElementById("pie"));
    this.setMainOption();
    this.setMainOptionPie();
  },
  computed: {
    fileNames() {
      if (this.files && this.files.length > 0)
        return this.files.map((e) => e.name);
      return [];
    },
    globalRate() {
      return this.data.map(
        (e) => parseFloat(e.reduce((accu, curr) => accu + curr, 0) / e.length).toFixed(2)
      );
    },
  },
  methods: {
    setMainOption() {
      let option = {
        title: {
          text: "Prediction CXR of COVID19",
          subtext: "VGG19",
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        legend: {
          data: this.labels,
        },
        toolbox: {
          show: true,
          feature: {
            mark: { show: true },
            dataView: { show: true, readOnly: false },
            magicType: { show: true, type: ["line", "bar"] },
            restore: { show: true },
            saveAsImage: { show: true },
          },
        },
        grid: {
          left: "3%",
          right: "4%",
          bottom: "3%",
          containLabel: true,
        },
        xAxis: {
          type: "value",
          boundaryGap: [0, 0.01],
        },
        yAxis: {
          type: "category",
          data: this.fileNames,
        },
        series: this.data.map((e, i) => ({
          name: this.labels[i],
          type: "bar",
          data: e,
        })),
      };

      option && this.main && this.main.setOption(option);
    },
    setMainOptionPie() {
      let option = {
        title: {
          text: "Prediction CXR of COVID19",
          subtext: "VGG19",
          left: "center",
        },
        tooltip: {
          trigger: "item",
        },
        legend: {
          left: "center",
          top: "bottom",
          data: this.labels,
        },
        labelLine: {
          show: false,
        },
        series: [
          {
            name: "访问来源",
            type: "pie",
            radius: ["40%", "70%"],
            avoidLabelOverlap: false,
            label: {
              show: false,
              position: "center",
            },
            emphasis: {
              label: {
                show: true,
                fontSize: "40",
                fontWeight: "bold",
              },
            },
            labelLine: {
              show: false,
            },
            data: this.globalRate.map((e, i) => ({
              value: e,
              name: this.labels[i],
            })),
          },
        ],
      };

      option && this.pie && this.pie.setOption(option);
    },
  },
  watch: {
    predectedList: {
      handler: function (val) {
        if (!(val && val.length)) return;
        this.data = val && val.length && val[0].map(() => []);
        val.forEach((e) => {
          e.forEach((v, i) => {
            this.data[i].push(parseFloat(v));
          });
        });
        this.setMainOption();
        this.setMainOptionPie();
      },
      immediate: true,
    },
  },
};
</script>

<style>
</style>