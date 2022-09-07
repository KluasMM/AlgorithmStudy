package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.Arrays;

/**
 * @Description
 * @Author T
 * @Date 2022/8/30
 */
@Service
public class PartEightGreedy {

    /**
     * 455. 分发饼干
     * <p>
     * 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
     * <p>
     * 对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。
     * 如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。
     * 你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
     *
     * @param g
     * @param s
     * @return
     */
    public int leetCode455(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int result = 0;
        int i = s.length - 1;
        int j = g.length - 1;

        while (i >= 0 && j >= 0) {
            if (s[i] >= g[j]) {
                i--;
                result++;
            }
            j--;
        }

        return result;
    }

    /**
     * 376. 摆动序列
     * <p>
     * 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。
     * 第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
     * <p>
     * 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。
     * <p>
     * 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，
     * 第二个序列是因为它的最后一个差值为零。
     * <p>
     * 子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
     * <p>
     * 给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度
     *
     * @param nums
     * @return
     */
    public int leetCode376(int[] nums) {
        /*
         * 解题思路：
         *  把数组想象成连绵起伏的山峰
         *  山峰和山谷代表着拐点 记录拐点数即可
         *  初始拐点为1 即数组最右侧直接记为一个拐点
         */
        //特判
        if (nums.length <= 1) return nums.length;

        //当前差值
        int curDiff;
        //上一组差值 初始值为0 即数组最左侧添加虚拟节点pre=nums[0]
        int preDiff = 0;
        //结果 数组最右侧直接记录为一个拐点
        int result = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            //计算当前差值
            curDiff = nums[i + 1] - nums[i];
            //当前差值和上一个差值互斥 则为拐点 preDiff=0的情况是为了最左侧结点记录
            if ((curDiff < 0 && preDiff >= 0) || (curDiff > 0 && preDiff <= 0)) {
                result++;
                preDiff = curDiff;
            }
        }

        return result;
    }
}
