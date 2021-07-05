<template>
  <el-dropdown :hide-on-click="false" class="center-dropdown">
    <span class="el-dropdown-link center-dropdown">
      Export<i class="el-icon-arrow-down el-icon--right"></i>
    </span>
    <el-dropdown-menu slot="dropdown">
      <el-dropdown-item>
        <export-excel
          :data="json_data"
          :fields="json_fields"
          worksheet="My Worksheet"
          name="Analysis_result_covid19.xls"
        >
          Excel
        </export-excel></el-dropdown-item
      >
      <el-dropdown-item>
        <export-excel
          :data="json_data"
          :fields="json_fields"
          type="csv"
          worksheet="My Worksheet"
          name="Analysis_result_covid19.xls"
        >
          csv
        </export-excel></el-dropdown-item
      >
    </el-dropdown-menu>
  </el-dropdown>
</template>

<script>
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
  computed: {
    json_fields() {
      let labels = {
        "File name": "file",
      };
      this.labels.forEach((label) => {
        labels[label] = label;
      });
      return labels;
    },
    json_data() {
      let data = [];
      const labels = Object.keys(this.json_fields);
      labels.shift();
      this.files.forEach((file, index) => {
        let predected = labels.reduce((accu, curr, i) => {
          accu[labels[i]] = this.predectedList[index][i];
          return accu;
        }, {});
        data.push({
          file: file.name,
          ...predected,
        });
      });
      return data;
    },
  },
  data: () => ({
    json_meta: [
      [
        {
          key: "charset",
          value: "utf-8",
        },
      ],
    ],
  }),
};
</script>

<style>
.center-dropdown {
  display: flex !important;
  align-items: center !important;
  cursor: pointer;
}
</style>