#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>  // nlohmann/json ���C�u�������C���N���[�h
#include <string>
#include <vector>
#include <Eigen/Dense>

using json = nlohmann::json;

struct PUZZLE {
    json j;
    int width;
    int height;
    std::vector<std::string> start;
    std::vector<std::string> goal;
    int n;
    std::vector<json> patterns;
    std::fstream fileStream;

    // �R���X�g���N�^�ŏ�����
    PUZZLE(const json& data) : j(data) {
        width = j["board"]["width"];
        height = j["board"]["height"];
        start = j["board"]["start"];
        goal = j["board"]["goal"];
        n = j["general"]["n"];
        patterns = j["general"]["patterns"];
    }
};

// �V�X�e���R�}���h�����s����֐�
void CMD(const char* cmdg_p) {
    system(cmdg_p);
}

void shiftRightPartial(Eigen::MatrixXi& mat, int rowIndex, int startCol, int shiftSize) {
    Eigen::VectorXi row = mat.block(rowIndex, startCol, 1, mat.cols() - startCol).transpose();
    std::rotate(row.data(), row.data() + row.size() - shiftSize, row.data() + row.size());
    mat.block(rowIndex, startCol, 1, mat.cols() - startCol) = row.transpose();
}


void shiftLeftPartial(Eigen::MatrixXi& mat, int rowIndex, int startCol, int shiftSize) {
    Eigen::VectorXi row = mat.block(rowIndex, startCol, 1, mat.cols() - startCol).transpose();
    std::rotate(row.data(), row.data() + shiftSize, row.data() + row.size());
    mat.block(rowIndex, startCol, 1, mat.cols() - startCol) = row.transpose();
}

void shiftUpPartial(Eigen::MatrixXi& mat, int colIndex, int startRow, int shiftSize) {
    Eigen::VectorXi column = mat.block(startRow, colIndex, mat.rows() - startRow, 1);
    std::rotate(column.data(), column.data() + shiftSize, column.data() + column.size());
    mat.block(startRow, colIndex, mat.rows() - startRow, 1) = column;
}


void shiftDownPartial(Eigen::MatrixXi& mat, int colIndex, int startRow, int shiftSize) {
    Eigen::VectorXi column = mat.block(startRow, colIndex, mat.rows() - startRow, 1);
    std::rotate(column.data(), column.data() + column.size() - shiftSize, column.data() + column.size());
    mat.block(startRow, colIndex, mat.rows() - startRow, 1) = column;
}

void shiftRightPartialb(Eigen::MatrixXi& mat, int rowIndex, int startCol, int shiftSize) {
    for (int i = 0; i < shiftSize; i++) {
        if (rowIndex + i < mat.rows() && startCol < mat.cols()) {
            Eigen::VectorXi row = mat.block(rowIndex + i, 0, 1, startCol + shiftSize).transpose();
            if (shiftSize < row.size()) {
                std::rotate(row.data(), row.data() + row.size() - shiftSize, row.data() + row.size());
                mat.block(rowIndex + i, 0, 1, startCol + shiftSize) = row.transpose();
            }
        }
    }
}

void shiftRightPartialc(Eigen::MatrixXi& mat, int rowIndex, int startCol, int shiftSize) {
    for (int i = 0; i < shiftSize; i++) {
        if (rowIndex + i < mat.rows() && startCol < mat.cols()) {
            Eigen::VectorXi row = mat.block(rowIndex + i, 0, 1, startCol+1).transpose();
            if (shiftSize < row.size()) {
                std::rotate(row.data(), row.data() + row.size() - shiftSize, row.data() + row.size());
                mat.block(rowIndex + i, 0, 1, startCol+1) = row.transpose();
            }
        }
    }
}

void shiftLeftPartialb(Eigen::MatrixXi& mat, int rowIndex, int startCol, int shiftSize) {
    for (int i = 0; i < shiftSize; ++i) {
        if (rowIndex + i < mat.rows() && startCol < mat.cols()) {
            Eigen::VectorXi row = mat.block(rowIndex + i, startCol, 1, mat.cols() - startCol).transpose();
            std::rotate(row.data(), row.data() + shiftSize, row.data() + row.size());
            mat.block(rowIndex + i, startCol, 1, mat.cols() - startCol) = row.transpose();
        }
    }
}

void shiftUpPartialb(Eigen::MatrixXi& mat, int colIndex, int startRow, int shiftSize) {
    for (int i = 0; i < shiftSize; i++) {
        if (colIndex + i < mat.cols() && startRow < mat.rows()) {
            Eigen::VectorXi column = mat.block(startRow, colIndex + i, mat.rows() - startRow, 1);
            std::rotate(column.data(), column.data() + shiftSize, column.data() + column.size());
            mat.block(startRow, colIndex + i, mat.rows() - startRow, 1) = column;
        }
    }
}


void shiftDownPartialb(Eigen::MatrixXi& mat, int colIndex, int startRow, int shiftSize) {

    for (int i = 0; i < shiftSize; i++) {
        if (colIndex + i < mat.cols() && startRow < mat.rows()) {
            Eigen::VectorXi column = mat.block(0, colIndex + i, startRow + shiftSize, 1);
            if (shiftSize < column.size()) {
                std::rotate(column.data(), column.data() + column.size() - shiftSize, column.data() + column.size());
                mat.block(0, colIndex + i, startRow + shiftSize, 1) = column;
            }
        }
    }
}



int valu(PUZZLE& puzzle, Eigen::MatrixXi now, Eigen::MatrixXi end)
{
    int valus = 0;
    Eigen::MatrixXi comparison = (now.array() == end.array()).cast<int>();
    valus = comparison.sum();  // ��v����v�f�̍��v
    return valus;
}

void Operate(PUZZLE& puzzle) {
    std::vector<std::string> save_bord;
    int count = 0;
    bool end = false;
    bool finish = false;
    int savescore = 0;

    Eigen::MatrixXi isBoard(puzzle.height, puzzle.width);
    Eigen::MatrixXi target(puzzle.height, puzzle.width);
    Eigen::MatrixXi save_Board(puzzle.height, puzzle.width);
    Eigen::MatrixXi great_Board(puzzle.height, puzzle.width);

    //�f�[�^��Eigen/Dense�ŗ��p�ł���`�ɕϊ�
    for (int i = 0; i < puzzle.height; ++i) {
        for (int j = 0; j < puzzle.width; ++j) {
            isBoard(i, j) = std::stoi(std::string(1, puzzle.start[i][j]));  // 1�����𐮐��ɕϊ�
            target(i, j) = std::stoi(std::string(1, puzzle.goal[i][j]));
        }
    }

    save_Board = isBoard;
    int i_save = 0;
    int j_save = 0;
    int s_save = 0;
    int score_save = 0;
    int oboerukun = -1;
    int kioku = 0;
    int num[] = { 16,8,4,2 };
    int pnum[] = { 10,7,4,1 };

    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> p;
    //��:0�C�� : 1�C�� : 2�C�E : 3
    std::vector<int> s;

    std::cout << "�͂���" << std::endl;
    std::cout << isBoard << std::endl << std::endl;


    /*while (count <= 10) {
        for (int pn = 0; pn < 4; pn++) {
            oboerukun = score_save;
            for (int i = 0; puzzle.height - num[pn] > i; i++) {
                for (int j = 0; puzzle.width - num[pn] > j; j++) {
                    for (int n = 0; n <= 3; n++) {
                        switch (n)
                        {
                        case 0:
                        {
                            shiftUpPartialb(save_Board, j, i, num[pn]);
                            if (score_save < valu(puzzle, save_Board, target)) {
                                i_save = i;
                                j_save = j;
                                s_save = 0;
                                great_Board = save_Board;
                                score_save = valu(puzzle, save_Board, target);

                            }
                            save_Board = isBoard;
                            break;

                        }
                        case 1:
                        {
                            shiftDownPartialb(save_Board, j, i, num[pn]);
                            if (score_save < valu(puzzle, save_Board, target)) {
                                i_save = i;
                                j_save = j;
                                s_save = 1;
                                great_Board = save_Board;
                                score_save = valu(puzzle, save_Board, target);

                            }
                            save_Board = isBoard;
                            break;
                        }
                        case 2:
                        {
                            shiftLeftPartialb(save_Board, i, j, num[pn]);
                            if (score_save < valu(puzzle, save_Board, target)) {
                                i_save = i;
                                j_save = j;
                                s_save = 2;
                                great_Board = save_Board;
                                score_save = valu(puzzle, save_Board, target);

                            }
                            save_Board = isBoard;
                            break;
                        }
                        case 3:
                        {
                            shiftRightPartialb(save_Board, i, j, num[pn]);
                            if (score_save < valu(puzzle, save_Board, target)) {
                                i_save = i;
                                j_save = j;
                                s_save = 3;
                                great_Board = save_Board;
                                score_save = valu(puzzle, save_Board, target);

                            }
                            save_Board = isBoard;
                            break;
                        }
                        }
                    }
                }

            }

            if (score_save != oboerukun) {
                std::cout << score_save << std::endl;
                y.push_back(i_save);
                x.push_back(j_save);
                s.push_back(s_save);
                p.push_back(pnum[pn]);
                count++;
                std::cout << "���ԔՖ�" << count << "���" << std::endl;
            }

            isBoard = save_Board = great_Board;

        }

    }*/



    

    while (isBoard != target) {

        for (int i = 0; puzzle.height > i; i++) {
            for (int j = 0; puzzle.width > j; j++) {
                //std::cout << isBoard << std::endl;
                if (isBoard(i, j) == target(i, j)) {
                    continue;
                }

                //�c�ɐ����̃}�X�����邩�T��
                for (int k = 1; puzzle.height - i > k; k++)
                {
                    if (target(i, j) == isBoard(i + k, j))
                    {

                        for (; k != 0;) {

                            if (k >= 256) {
                                shiftUpPartialb(isBoard, j, i, 256);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(22);
                                count++;

                                k -= 256;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 128) {
                                shiftUpPartialb(isBoard, j, i, 128);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(19);
                                count++;

                                k -= 128;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 64) {
                                shiftUpPartialb(isBoard, j, i, 64);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(16);
                                count++;

                                k -= 64;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 32) {
                                shiftUpPartialb(isBoard, j, i, 32);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(13);
                                count++;

                                k -= 32;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 16) {
                                shiftUpPartialb(isBoard, j, i, 16);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(10);
                                count++;

                                k -= 16;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 8) {
                                shiftUpPartialb(isBoard, j, i, 8);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(7);
                                count++;

                                k -= 8;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 4) {
                                shiftUpPartialb(isBoard, j, i, 4);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(4);
                                count++;

                                k -= 4;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 2) {
                                shiftUpPartialb(isBoard, j, i, 2);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(1);
                                count++;

                                k -= 2;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else {
                                shiftUpPartial(isBoard, j, i, 1);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(0);
                                count++;

                                k--;
                                std::cout << count << "�c���" << std::endl;

                            }
                        }
                        end = true;
                        break;
                    }

                }
                if (end) {
                    end = false;
                    continue;
                }

                //���ɐ����̃}�X�����邩�T��
                for (int k = 1; puzzle.width - j > k; k++) {
                    if (target(i, j) == isBoard(i, j + k)) {
                        for (; k != 0;) {
                            if (k >= 256) {
                                shiftLeftPartialb(isBoard, i, j, 256);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(22);
                                count++;
                                k -= 256;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 128) {
                                shiftLeftPartialb(isBoard, i, j, 128);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(19);
                                count++;
                                k -= 128;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 64) {
                                shiftLeftPartialb(isBoard, i, j, 64);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(16);
                                count++;
                                k -= 64;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 32) {
                                shiftLeftPartialb(isBoard, i, j, 32);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(13);
                                count++;
                                k -= 32;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 16) {
                                shiftLeftPartialb(isBoard, i, j, 16);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(10);
                                count++;
                                k -= 16;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 8) {
                                shiftLeftPartialb(isBoard, i, j, 8);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(7);
                                count++;
                                k -= 8;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 4) {
                                shiftLeftPartialb(isBoard, i, j, 4);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(4);
                                count++;
                                k -= 4;
                                std::cout << count << "�����" << std::endl;
                            }
                            else if (k >= 2) {
                                shiftLeftPartialb(isBoard, i, j, 2);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(1);
                                count++;
                                k -= 2;
                                std::cout << count << "�����" << std::endl;
                            }
                            else {
                                shiftLeftPartialb(isBoard, i, j, 1);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(2);
                                p.push_back(0);
                                count++;
                                k--;
                                std::cout << count << "�����" << std::endl;
                            }
                        }
                        end = true;
                        break;
                    }

                }

                if (end) {
                    end = false;
                    continue;
                }


                //�c���Ɉ�v����}�X���Ȃ������ꍇ�c�̃}�X�Ɉړ�������
                while (!end) {

                    for (int a = 1; a < puzzle.height - i; a++) {
                        for (int b = 0; b < puzzle.width; b++) {
                            if (target(i, j) == isBoard(a + i, b)) {
                                int jb = j - b;
                                int bj = b - j;

                                if (jb >= 1) {
                                    for (; jb != 0;) {
                                        if (jb >= 256) {
                                            shiftRightPartialc(isBoard, i + a, j, 256);
                                            y.push_back(i + a);
                                            //j-shiftsize+1�̒l�ɔ��^��K�p����
                                            x.push_back(j - 255);
                                            s.push_back(3);
                                            p.push_back(22);
                                            count++;
                                            jb -= 256;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 128) {
                                            shiftRightPartialc(isBoard, i + a, j, 128);
                                            y.push_back(i + a);
                                            //j-shiftsize+1�̒l�ɔ��^��K�p����
                                            x.push_back(j - 127);
                                            s.push_back(3);
                                            p.push_back(19);
                                            count++;
                                            jb -= 128;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 64) {
                                            shiftRightPartialc(isBoard, i + a, j, 64);
                                            y.push_back(i + a);
                                            //j-shiftsize+1�̒l�ɔ��^��K�p����
                                            x.push_back(j - 63);
                                            s.push_back(3);
                                            p.push_back(16);
                                            count++;
                                            jb -= 64;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 32) {
                                            shiftRightPartialc(isBoard, i + a, j, 32);
                                            y.push_back(i + a);
                                            //j-shiftsize+1�̒l�ɔ��^��K�p����
                                            x.push_back(j - 31);
                                            s.push_back(3);
                                            p.push_back(13);
                                            count++;
                                            jb -= 32;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 16) {
                                            shiftRightPartialc(isBoard, i + a, j, 16);
                                            y.push_back(i + a);
                                            //j-shiftsize+1�̒l�ɔ��^��K�p����
                                            x.push_back(j - 15);
                                            s.push_back(3);
                                            p.push_back(10);
                                            count++;
                                            jb -= 16;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 8) {
                                            shiftRightPartialc(isBoard, i + a, j, 8);
                                            y.push_back(i + a);
                                            x.push_back(j - 7);
                                            s.push_back(3);
                                            p.push_back(7);
                                            count++;
                                            jb -= 8;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 4) {
                                            shiftRightPartialc(isBoard, i + a, j, 4);
                                            y.push_back(i + a);
                                            x.push_back(j - 3);
                                            s.push_back(3);
                                            p.push_back(4);
                                            count++;
                                            jb -= 4;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (jb >= 2) {
                                            shiftRightPartialc(isBoard, i + a, j, 2);
                                            y.push_back(i + a);
                                            x.push_back(j - 1);
                                            s.push_back(3);
                                            p.push_back(1);
                                            count++;
                                            jb -= 2;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else {
                                            shiftRightPartialc(isBoard, i + a, j, 1);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(3);
                                            p.push_back(0);
                                            count++;
                                            jb--;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                    }
                                    end = true;
                                    break;
                                }
                                else if (bj >= 1) {

                                    for (; bj != 0;) {
                                        if (bj >= 256) {
                                            shiftLeftPartialb(isBoard, i + a, j, 256);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(22);
                                            count++;
                                            bj -= 256;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 128) {
                                            shiftLeftPartialb(isBoard, i + a, j, 128);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(19);
                                            count++;
                                            bj -= 128;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 64) {
                                            shiftLeftPartialb(isBoard, i + a, j, 64);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(16);
                                            count++;
                                            bj -= 64;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                         else if (bj >= 32) {
                                            shiftLeftPartialb(isBoard, i + a, j, 32);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(13);
                                            count++;
                                            bj -= 32;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 16) {
                                            shiftLeftPartialb(isBoard, i + a, j, 16);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(10);
                                            count++;
                                            bj -= 16;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 8) {
                                            shiftLeftPartialb(isBoard, i + a, j, 8);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(7);
                                            count++;
                                            bj -= 8;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 4) {
                                            shiftLeftPartialb(isBoard, i + a, j, 4);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(4);
                                            count++;
                                            bj -= 4;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else if (bj >= 2) {
                                            shiftLeftPartialb(isBoard, i + a, j, 2);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(1);
                                            count++;
                                            bj -= 2;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                        else {
                                            shiftLeftPartial(isBoard, i + a, j, 1);
                                            y.push_back(i + a);
                                            x.push_back(j);
                                            s.push_back(2);
                                            p.push_back(0);
                                            count++;
                                            bj--;
                                            std::cout << count << "�����" << std::endl;
                                        }
                                    }
                                    end = true;
                                    break;
                                }
                            }
                        }
                        if (end)
                            break;
                    }
                    if (end) {
                        end = false;
                        break;
                    }
                }

                //�c�ɐ����̃}�X�����邩�T��
                for (int k = 1; puzzle.height - i > k; k++)
                {
                    if (target(i, j) == isBoard(i + k, j))
                    {

                        for (; k != 0;) {

                            if (k >= 256) {
                                shiftUpPartialb(isBoard, j, i, 256);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(22);
                                count++;

                                k -= 256;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 128) {
                                shiftUpPartialb(isBoard, j, i, 128);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(19);
                                count++;

                                k -= 128;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 64) {
                                shiftUpPartialb(isBoard, j, i, 64);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(16);
                                count++;

                                k -= 64;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 32) {
                                shiftUpPartialb(isBoard, j, i, 32);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(13);
                                count++;

                                k -= 32;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 16) {
                                shiftUpPartialb(isBoard, j, i, 16);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(10);
                                count++;

                                k -= 16;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 8) {
                                shiftUpPartialb(isBoard, j, i, 8);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(7);
                                count++;

                                k -= 8;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 4) {
                                shiftUpPartialb(isBoard, j, i, 4);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(4);
                                count++;

                                k -= 4;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else if (k >= 2) {
                                shiftUpPartialb(isBoard, j, i, 2);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(1);
                                count++;

                                k -= 2;
                                std::cout << count << "�c���" << std::endl;

                            }
                            else {
                                shiftUpPartial(isBoard, j, i, 1);
                                y.push_back(i);
                                x.push_back(j);
                                s.push_back(0);
                                p.push_back(0);
                                count++;

                                k--;
                                std::cout << count << "�c���" << std::endl;

                            }
                        }
                        end = true;
                        break;
                    }

                }
                if (end) {
                    end = false;
                    continue;
                }

            }

        }

    }

   

    // JSON�I�u�W�F�N�g���t�@�C���ɏ������݂܂�
    std::ofstream fileStream("output.json");
    if (fileStream.is_open()) {

        fileStream << "{" << std::endl << "\"n\":" << count << "," << std::endl << "\"ops\":[" << std::endl;
        for (int n = 0; n < count; n++) {
            fileStream << "{" << std::endl << "\"p\":" << p[n] << ",\"x\":" << x[n] << ",\"y\":" << y[n] << ",\"s\":" << s[n] << std::endl << "}";
            if (n < count - 1) {
                fileStream << "," << std::endl; // �Ō�̗v�f�ɂ̓J���}��t���Ȃ�
            }
            fileStream << std::endl;
        }
        fileStream << "]" << std::endl << "}" << std::endl;

        fileStream.close();
        std::cout << "JSON�t�@�C�����쐬����܂����B" << std::endl;
    }
    else {
        std::cerr << "�t�@�C�����J�����Ƃ��ł��܂���ł����B" << std::endl;
    }

    const char postcode[] = "curl -X POST -H \"Content-Type:application/json\" -H \"Procon-Token: token1\" localhost:8080/answer -d @output.json";
    CMD(postcode); // JSON�t�@�C���𑗂�

    for (int i = 0; puzzle.height > i; i++) {
        std::cout << "Start: " << isBoard.row(i) << std::endl;
        std::cout << "Goal:  " << target.row(i) << std::endl;
    }
    std::cout << "�Ō�̕]���l:" << valu(puzzle, isBoard, target) << std::endl;
    std::cout << "Start: " << std::endl << isBoard  << std::endl<< std::endl << std::endl;
    std::cout << "Goal: " << std::endl << target << std::endl << std::endl;

    
}

int main(void) {

    const char cmdg[] = "curl -v localhost:8080/problem -H \"Procon-Token: token1\" -o input.json";
    CMD(cmdg); // JSON�t�@�C�����󂯎��

    // JSON�t�@�C����ǂݍ���
    std::ifstream file("input.json");  // JSON�f�[�^���܂܂�Ă���t�@�C�������w��
    if (!file.is_open()) {
        std::cerr << "�t�@�C�����J���܂���ł����B" << std::endl;
        return 1;
    }

    json j;
    file >> j;
    file.close();

    // PUZZLE�I�u�W�F�N�g�̍쐬
    PUZZLE puzzle(j);
    //READ(puzzle);s
    Operate(puzzle);

    return 0;
}